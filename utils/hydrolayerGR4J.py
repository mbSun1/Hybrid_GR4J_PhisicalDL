import math

import torch
import torch.nn as nn


class PRNNLayer(nn.Module):
    """Implementation of the standard P-RNN layer
    Hyper-parameters
    ----------
    mode: if in "normal", the output will be the generated flow;
          if in "analysis", the output will be a tensor containing all state variables and process variables
    ==========
    Parameters
    ----------
    x1: maximum capacity of the production storage (mm) | range: (1, 2000)
    x2: groundwater exchange coefficient (mm) | range: (-20, 20)
    x3: maximum capacity of the routing storage (mm) | range: (1, 300)
    x4: time base of the unit hydrograph UH1 (days) | range: (0.5, 15)
    """

    def __init__(self,  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), mode='normal', **kwargs):
        self.mode = mode
        super(PRNNLayer, self).__init__(**kwargs)
        self.hydropara = nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5]).to(device))  # x1, x2, x3, x4
        # 变换参数区间
        self.x1 = self.hydropara[0].item() * (2000-1) + 1  # self.x1∈(1.0, 2000)
        self.x2 = self.hydropara[1].item() * (20-(-20)) - 20  # self.x2∈(-20, 20)
        self.x3 = self.hydropara[2].item() * (300-1) + 1  # self.x3∈(1, 300)
        self.x4 = self.hydropara[3].item() * (30-1) + 1  # self.x4∈(1, 30)

    def heaviside(self, x):
        """
        A smooth approximation of Heaviside step function
            if x < 0: heaviside(x) ~= 0
            if x > 0: heaviside(x) ~= 1
        """

        return (torch.tanh(5 * x) + 1) / 2

    def netPrec_Etp(self, prcp, pet, prod_s, x1):
        """
        Net precipitation and net evaporation calculation module, before entering the runoff production module
        If prcp > pet: there is no evaporation in the runoff storage, e_s = 0
        If prcp < pet: no rainwater is allocated to the runoff storage, p_s = 0
        """
        dimension_sizes = prcp.size()[0]
        p_n_list = []
        pe_n_list = []
        p_s_list = []
        e_s_list = []
        for i in range(dimension_sizes):
            #Take out the rainfall, evaporation, and runoff state at each step
            prcp_step = prcp[i:(i + 1), :].item()
            pet_step = pet[i:(i + 1), :].item()
            prod_s_step = prod_s[i:(i + 1), :].item()

            # 英文翻译：Calculate net precipitation
            p_n_step, pe_n_step, p_s_step, e_s_step = 0, 0, 0, 0
            if prcp_step >= pet_step:
                # Calculate net precipitation
                p_n_step = prcp_step - pet_step
                # Calculate net evaporation
                pe_n_step = 0

                #Rainwater allocated to production storage
                p_s_step = ((x1 * (1 - (prod_s_step / x1) ** 2) * math.tanh(p_n_step / x1)) /
                            (1 + prod_s_step / x1 * math.tanh(p_n_step / x1)))
                #The part evaporated from the runoff storage：
                e_s_step = 0
            else:
                p_n_step = 0
                pe_n_step = pet_step - prcp_step
                e_s_step = ((prod_s_step * (2 - prod_s_step / x1) * math.tanh(pe_n_step / x1))
                            / (1 + (1 - prod_s_step / x1) * math.tanh(pe_n_step / x1)))
                p_s_step = 0

            #  Put the net precipitation, net evaporation, rainwater allocated to the production storage, and the part evaporated from the runoff storage at each step into the list
            p_n_list.append(p_n_step)
            pe_n_list.append(pe_n_step)
            p_s_list.append(p_s_step)
            e_s_list.append(e_s_step)

        p_n = torch.unsqueeze(torch.tensor(p_n_list), dim=1)
        pe_n = torch.unsqueeze(torch.tensor(pe_n_list), dim=1)
        p_s = torch.unsqueeze(torch.tensor(p_s_list), dim=1)
        e_s = torch.unsqueeze(torch.tensor(e_s_list), dim=1)

        return [p_n, pe_n, p_s, e_s]

    def productionStorage(self, prod_s, prcp, pet, x1):

        # Calculate net precipitation and net evaporation
        [p_n, pe_n, p_s, e_s] = self.netPrec_Etp(prcp, pet, prod_s, self.x1)

        dimension_sizes = pe_n.size()[0]
        p_r_list = []
        next_prod_s_list = []
        for i in range(dimension_sizes):
            # Take out the production storage state, net precipitation, net evaporation, rainwater allocated to the production storage, and the part evaporated from the production storage at each step
            prod_s_step = prod_s[i:(i + 1), :].item()
            p_n_step = p_n[i:(i + 1), :].item()
            p_s_step = p_s[i:(i + 1), :].item()
            e_s_step = e_s[i:(i + 1), :].item()

            #Calculate the new production storage state of this step (that is, the initial production storage state of the next step)
            next_prod_s_step = prod_s_step - e_s_step + p_s_step

            # Calculate the actual percolation amount at this time step
            perc_step = next_prod_s_step * (1 - (1 + (4 / 9 * next_prod_s_step / x1) ** 4) ** (-0.25))

            # Calculate the new production storage state of this step (that is, the initial production storage state of the next step)
            next_prod_s_step = next_prod_s_step - perc_step

            # The total amount of water reaching the nonlinear reservoir confluence storage at this time step
            p_r_step = perc_step + (p_n_step - p_s_step)

            #  Put the total amount of water reaching the nonlinear reservoir confluence storage and the production storage state at each step into the list
            p_r_list.append(p_r_step)
            next_prod_s_list.append(next_prod_s_step)

        p_r = torch.unsqueeze(torch.tensor(p_r_list), dim=1)
        next_prod_s = torch.unsqueeze(torch.tensor(next_prod_s_list), dim=1)

        return [p_r, next_prod_s]

    def routingStorage(self, rout_s, p_r, x2, x3, x4):
        """
            Routing calculation module
        """
        dimension_sizes = p_r.size()[0]
        qsim_list = []
        next_rout_s_list = []
        for i in range(dimension_sizes):
            # Take out the nonlinear reservoir confluence storage state, the total amount of water reaching the nonlinear reservoir confluence storage, and the net evaporation at each step
            rout_s_step = rout_s[i:(i + 1), :].item()
            p_r_step = p_r[i:(i + 1), :].item()

            # Calculate the infiltration and net precipitation entering the routing storage
            # Distribute this water to different routing paths (UH1 and UH2) in a ratio of 0.9/0.1
            p_r_uh1 = 0.9 * p_r_step
            p_r_uh2 = 0.1 * p_r_step

            # Calculate the unit hydrograph order
            num_uh1 = int(math.ceil(x4))
            num_uh2 = int(math.ceil(2 * x4 + 1))

            # Calculate the ordinate of the two unit hydrographs
            uh1_ordinates = [0] * num_uh1
            uh2_ordinates = [0] * num_uh2

            for j in range(1, num_uh1 + 1):
                uh1_ordinates[j - 1] = self.s_curve1(j, x4) - self.s_curve1(j - 1, x4)

            for j in range(1, num_uh2 + 1):
                uh2_ordinates[j - 1] = self.s_curve2(j, x4) - self.s_curve2(j - 1, x4)

            # Store the rainwater through the unit hydrograph distribution
            uh1 = [0] * num_uh1
            uh2 = [0] * num_uh2

            #  Update the confluence state through the unit hydrograph distribution
            for j in range(0, num_uh1 - 1):
                uh1[j] = uh1[j + 1] + uh1_ordinates[j] * p_r_uh1
            uh1[-1] = uh1_ordinates[-1] * p_r_uh1

            for j in range(0, num_uh2 - 1):
                uh2[j] = uh2[j + 1] + uh2_ordinates[j] * p_r_uh2
            uh2[-1] = uh2_ordinates[-1] * p_r_uh2

            # Calculate groundwater exchange F
            gw_exchange = x2 * (rout_s_step / x3) ** 3.5

            # Update confluence storage
            next_rout_s_step = max(0, rout_s_step + uh1[0] + gw_exchange)
            # Confluence storage outflow
            q_r_step = next_rout_s_step * (1 - (1 + (next_rout_s_step / x3) ** 4) ** (-0.25))

            # Subtract outflow from confluence storage
            next_rout_s_step = next_rout_s_step - q_r_step

            # Calculate the flow component of unit hydrograph 2
            q_d_step = max(0, uh2[0] + gw_exchange)

            # Total runoff in this time period
            qsim_step = q_r_step + q_d_step

            # Put the total runoff and the nonlinear reservoir confluence state at each step into the list
            qsim_list.append(qsim_step)
            next_rout_s_list.append(next_rout_s_step)

        # Format conversion
        qsim = torch.unsqueeze(torch.tensor(qsim_list), dim=1)
        next_rout_s_step = torch.unsqueeze(torch.tensor(next_rout_s_list), dim=1)

        return [qsim, next_rout_s_step]

    def s_curve1(self, t, x4):
        """Calculate the s-curve of the unit-hydrograph 1.

        Args:
            t: timestep
            x4: model parameter x4 of the gr4j model.

        """
        if t <= 0:
            return 0.
        elif t < x4:
            return (t / x4) ** 2.5
        else:
            return 1.

    def s_curve2(self, t, x4):
        """Calculate the s-curve of the unit-hydrograph 2.

        Args:
            t: timestep
            x4: model parameter x4 of the gr4j model.
        """

        if t <= 0:
            return 0.
        elif t <= x4:
            return 0.5 * ((t / x4) ** 2.5)
        elif t < 2 * x4:
            return 1 - 0.5 * ((2 - t / x4) ** 2.5)
        else:
            return 1.

    def prnn_cell(self, step_in, H0):  # Define step function for the RNN
        """
        :param step_in: (N, H_in)
        :param H0: (N, H_out * 2)
        """
        prod_s = H0[0]  # prod_s
        rout_s = H0[1]  # rout_s

        # Load the current input column
        p = step_in[:, 0:1]#(5,1)
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]

        #1.Production module
        #  Calculate net precipitation and net evaporation, the initial production storage of the next step next_prod_s, and the total amount of water reaching the nonlinear reservoir confluence storage pr
        [p_r, _next_prod_s] = self.productionStorage(prod_s, p, pet, self.x1)

        #2.Routing module
        #Perform routing calculation on the infiltration and net precipitation pr entering the routing storage
        [qsim,_next_rout_s]=self.routingStorage(rout_s, p_r, self.x2, self.x3, self.x4)

        # Record all state variables that depend on the previous step
        next_prod_s = prod_s + torch.clip(_next_prod_s, -1e5, 1e5)
        next_rout_s = rout_s + torch.clip(_next_rout_s, -1e5, 1e5)

        return qsim, tuple((next_prod_s, next_rout_s))

    def forward(self, inputs, hidden, batch_first=True):
        # Load the input vector
        batch_dim = 0 if batch_first else 1
        sequence_dim = 1 if batch_first else 0
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(batch_dim)
        prcp = inputs[:, :, 0:1]#(5,1825,1)
        tmean = inputs[:, :, 1:2]#(5,1825,1)
        # pet = inputs[:, :, 2:3]

        # Calculate PET using Hamon’s formulation
        dayl = inputs[:, :, 2:3]#(5,1825,1)
        pet = 29.8 * (dayl * 24) * 0.611 * torch.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)#(5,1825,1)

        # Concatenate prcp, tmean, and pet into a new input
        new_inputs = torch.concat((prcp, tmean, pet), axis=-1)#(5,1825,3)

        # for the last batch, the first dim may be smaller than batch_size
        batch_size = new_inputs.size(batch_dim)
        hidden = (hidden[0][:batch_size], hidden[1][:batch_size])

        # Recursively calculate state variables by using RNN #flow(5,1825,1).states(5,1825,1)
        flow, states = None, None
        for i in range(new_inputs.size(sequence_dim)):
            states = torch.cat(hidden, axis=1).unsqueeze(1) if states is None else \
                torch.concat((states, torch.cat(hidden, axis=1).unsqueeze(1)), axis=1)
            output, hidden = self.prnn_cell(new_inputs[:, i], hidden)
            flow = output.unsqueeze(1) if flow is None else torch.concat((flow, output.unsqueeze(1)), axis=1)

        if self.mode == "normal":
            return flow, hidden

