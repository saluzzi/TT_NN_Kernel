import numpy as np
import scipy
import datetime





class ValueFunction():
    def __init__(self, t0, t1, dt, dx, gamma, P_sdre, Ax, gradv=None, str_control='sdre', flag_verbose=False):
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.dx = dx
        self.gamma = gamma

        self.P_sdre = P_sdre
        self.Ax = Ax
        
        
        self.str_control = str_control
        self.gradv = gradv


        assert str_control in ['sdre', 'surr', 'zero'], 'str_control must be either sdre, surr or zero!'

        if str_control == 'surr':
            assert gradv is not None, 'If str_control==surr, then gradv must be provided!'

        self.flag_verbose = flag_verbose

    def control_u(self, x):
        if self.str_control == 'sdre':         # use surrogate
            if np.linalg.norm(x) < 3e-1:                # 3e-1 for kernel, 8e-1 for NN
                ret = -self.P_sdre(np.zeros(x.shape)) @ x / self.gamma
            else:
                ret = -self.P_sdre(x) @ x / self.gamma
        elif self.str_control == 'surr':
            if np.linalg.norm(x) < 3e-1:                # 3e-1 for kernel, 8e-1 for NN
                ret = -self.P_sdre(np.zeros(x.shape)) @ x / self.gamma
            else:
                ret = -self.gradv(x).reshape(-1) / (2*self.gamma)
        elif self.str_control == 'zero':
            ret = np.zeros(x.shape)

        return ret

    
    def costs(self, x):
        return self.dx * np.sum(x**2) + self.gamma * np.sum( (self.P_sdre(x) @ x / self.dx)**2 )
    
    def value_function_mathias(self, x):

        integrat = scipy.integrate.ode(self.dynamics_func).set_integrator('vode', method='bdf')
        integrat.set_initial_value(x, self.t0)

        cost = self.dt * self.costs(x)/2

        idx_counter = 0

        list_y = []

        while integrat.successful() and integrat.t < self.t1:
            integrat.integrate(integrat.t + self.dt)
            cost+=self.dt * self.costs(integrat.y)

            gradient = self.control_u(integrat.y)


            list_y.append(integrat.y)

            idx_counter += 1
            if idx_counter % 10 == 0 and self.flag_verbose:
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                  'time = {:.3f}, norm-gradient = {:.3e}, norm-x = {:.3e}'.format(
                      idx_counter * self.dt, np.linalg.norm(gradient), np.linalg.norm(integrat.y)))

            # print(integrat.successful())

        cost = cost-self.dt * self.costs(integrat.y)/2

        print(idx_counter)

        return cost, list_y
    

    def dynamics_func(self, t, x):
        ret = self.Ax(x) @ x + self.control_u(x)

        return ret
    
    def value_function_luca(self, x):

        n_timesteps = int((self.t1 - self.t0 ) / self.dt)
        t_control = np.linspace(0, self.t1, n_timesteps)  # Adjust the number of time steps as needed

        y_control = scipy.integrate.odeint(self.dynamics_func, x, t_control)

        cost = np.zeros(n_timesteps)
        opt_con = np.zeros((x.shape[1], n_timesteps))

        # Compute cost values along trajectory
        for idx_timestep in range(n_timesteps):
            opt_con[:, idx_timestep] = self.control_u(y_control[idx_timestep])
            cost[idx_timestep] = self.dx * np.sum(y_control[idx_timestep]**2) \
                + self.gamma * np.sum(opt_con[:, idx_timestep]**2)

        # Use trapezoidal rule to compute the overall cost
        total_cost = np.sum((t_control[1:] - t_control[:-1]) * (cost[1:] + cost[:-1])) / 2

        return total_cost, y_control, t_control, opt_con

        
