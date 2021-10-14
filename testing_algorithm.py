from quadrotor import AgentQuadrotor
from nnet import NNet
from typing import List
import numpy as np
import os
import copy
from enum import IntEnum
import matplotlib.pyplot as plt 

class RA(IntEnum):
    NA = -1
    COC = 0
    WL = 1
    WR = 2
    SL = 3
    SR = 4


class TestCase:
    def __init__(self, idx, theta, vint, rho, h, time2collision, timehorizon):
        self.theta = theta 
        self.rho = rho 
        self.time2collision = time2collision
        self.timehorizon = timehorizon
        self.simulation_time = 0
        self.idx = idx
        self.h = h
        scale = 0.3048 / 40
        self.valid = False
        # scale = 1
        T = self.time2collision
        T_wp = self.timehorizon

        init_pos_0 = [0, 0, 60]
        init_yaw_0 = np.pi / 2

        if theta>0 and theta<np.pi:
            phi = -np.arcsin(rho*np.sin(abs(theta))/(T*vint))
            psi = np.pi-abs(theta)-abs(phi)
            vown = vint*np.sin(psi)/np.sin(abs(theta))
        elif theta>-np.pi and theta<0:
            phi = np.arcsin(rho*np.sin(abs(theta))/(T*vint))
            psi = np.pi-abs(theta)-abs(phi)
            vown = vint*np.sin(psi)/np.sin(abs(theta))
        elif theta==abs(np.pi):
            phi = 0
            psi = 0
            vown = (T*vint-rho)/T
        else:
            phi = 0
            psi = 0
            vown = (T*vint+rho)/T 

        if np.isnan(phi) or np.isnan(psi) or np.isnan(vown):
            print(f'Configuration {theta}, {vint}, {rho} is not available')
            return 
        xint = init_pos_0[0]+rho*np.cos(theta+np.pi/2)
        yint = init_pos_0[1]+rho*np.sin(theta+np.pi/2)

        init_lin_vel_0 = [0, vown * scale, 0]
        init_pos_1 = [xint * scale, yint * scale, init_pos_0[2] - h*scale]
        init_yaw_1 = np.pi / 2 + phi
        init_lin_vel_1 = [vint * np.cos(init_yaw_1) * scale, vint * np.sin(init_yaw_1) * scale, h/T*scale]
    
        init_0 = init_pos_0 + init_lin_vel_0 + [0,0,0] + [0,0,0]
        init_1 = init_pos_1 + init_lin_vel_1 + [0,0,0] + [0,0,0]
        init = [init_0, init_1]

        self.desired_speed = [vown*scale, vint*scale] 
        
        waypoints_dict = {}
        waypoints_dict['drone0'] =[]
        waypoints_dict['drone1'] =[]
        self.waypoints_list = [
            (init_pos_0[0] + init_lin_vel_0[0] * (0+1) * T_wp, init_pos_0[1] + init_lin_vel_0[1] * (0+1) * T_wp, init_pos_0[2]), 
            (init_pos_1[0] + init_lin_vel_1[0] * (0+1) * T_wp, init_pos_1[1] + init_lin_vel_1[1] * (0+1) * T_wp, init_pos_1[2] + init_lin_vel_1[2] * (0+1) * T_wp)
        ]
        
        self.trajectory_dict = {
            0: [[init_0,(),self.idx,0,RA.NA, 0]],
            1: [[init_1,(),self.idx,1,RA.NA, 0]]
        }
        self.valid = True

    def match(self, agent_idx) -> bool:
        trajectory0 = self.trajectory_dict[0]
        trajectory1 = self.trajectory_dict[1]
        if agent_idx == 0:
            trajectory0_len = len(trajectory0)
            if len(trajectory0) <= len(trajectory1) and trajectory0[trajectory0_len-1][1] == () and trajectory1[trajectory0_len-1][1] == ():
                return True
            else:
                return False
        elif agent_idx == 1:
            trajectory1_len = len(trajectory1)
            if len(trajectory1) <= len(trajectory0) and trajectory1[trajectory1_len-1][1] == () and trajectory0[trajectory1_len-1][1] == ():
                return True
            else:
                return False
        return False

class NewAlgorithm:
    def __init__(self, test_suite):
        self.test_stack = []
        self.test_cases: List[TestCase] = self.process_test_cases(test_suite)
        self.acas_networks = self.load_networks('./nnet')
        self.cache_dict = {}
        self.agent = AgentQuadrotor(0, [], [0,0,0,0,0,0,0,0,0,0,0,0])
        self.resolution = [0.001,0.001,0.001]

    def process_test_cases(self,test_suite):
        res = []
        i = 0
        for test in test_suite:
            theta = test[0]
            vint = test[1]
            rho = test[2]
            h = test[3]
            T = test[4]
            T_wp = test[5]
            test_case = TestCase(i, theta, vint, rho, h, T, T_wp)
            if test_case.valid:
                i += 1
                res.append(test_case)
        return res 

    def generate_acas(self, ownship_state, intruder_state, tau, prev_acas):
        ra = RA.NA
        if prev_acas == RA.NA:
            return RA.NA
        key = (int(prev_acas), tau)
        network: NNet = self.acas_networks[key]
        ownship_array = np.array(ownship_state)
        intruder_array = np.array(intruder_state)

        rho = np.linalg.norm(ownship_array[:2] - intruder_array[:2])
        phi0 = np.arctan2(ownship_array[4], ownship_array[3])
        phi1 = np.arctan2(intruder_array[4], intruder_array[3])
        phi = phi1 - phi0 
        theta = 2*np.arctan2(intruder_array[1] - ownship_array[1], intruder_array[0] - ownship_array[0] + rho) - phi0
        v0 = np.linalg.norm(ownship_array[3:5])
        v1 = np.linalg.norm(intruder_array[3:5])

        network_input = [rho, theta, phi, v0, v1]
        res = network.evaluate_network(np.array(network_input))
        ra = RA(np.argmax(res))
        return ra

    def process_acas(self, ra, agent_state, test_idx, agent_idx):
        desired_velocity = ()
        horiz_spd = np.linalg.norm(np.array(agent_state[3:5]))
        phi = np.arctan2(agent_state[4], agent_state[3])
        wp = self.test_cases[test_idx].waypoints_list[agent_idx]
        v = self.test_cases[test_idx].desired_speed[agent_idx]
        if ra == RA.COC:
            # Going towards the goal at veocities specified at begining 
            x = agent_state[0]
            y = agent_state[1]
            ori = np.arctan2(wp[1] - y, wp[0] - x)
            desired_velocity = (np.cos(ori)*horiz_spd,np.sin(ori)*horiz_spd,0)
        elif ra == RA.SL:
            # Strong turn to left at velocities specified at begining 
            ori = phi - 3*np.pi/180
            desired_velocity = (np.cos(ori)*horiz_spd,np.sin(ori)*horiz_spd,0)
        elif ra == RA.WL:
            # Weak turn to left at velocities specified at begining 
            ori = phi - 1.5*np.pi/180
            desired_velocity = (np.cos(ori)*horiz_spd,np.sin(ori)*horiz_spd,0)
        elif ra == RA.SR:
            # Strong turn to right at velocities specified at begining 
            ori = phi + 3*np.pi/180
            desired_velocity = (np.cos(ori)*horiz_spd,np.sin(ori)*horiz_spd,0)
        elif ra == RA.WR:
            # Weak turn to right at velocities specified at begining 
            ori = phi + 1.5*np.pi/180
            desired_velocity = (np.cos(ori)*horiz_spd,np.sin(ori)*horiz_spd,0)
        else:
            # Same as RA.COC
            x = agent_state[0]
            y = agent_state[1]
            ori = np.arctan2(wp[1] - y, wp[0] - x)
            desired_velocity = (np.cos(ori)*horiz_spd,np.sin(ori)*horiz_spd,0)
        return desired_velocity

    def load_networks(self, folder):
        path = os.path.abspath(folder)
        networks = {}
        for prev_acas in range(5):
            for tau in range(9):
                fn = path + f"/ACASXU_run2a_{prev_acas+1}_{tau+1}_batch_2000.nnet"
                net = NNet(fn)
                networks[(prev_acas,tau)] = net
        return networks

    def get_new_xp_pair(self, test_idx: int, agent_idx: int):
        test_case: TestCase = self.test_cases[test_idx]
        
        # Obtain current states
        idx = len(test_case.trajectory_dict[agent_idx])-1
        xp0 = test_case.trajectory_dict[0][idx]
        xp1 = test_case.trajectory_dict[1][idx]
        agent0_state = xp0[0]
        agent1_state = xp1[0]

        # Obtain previous ACAS
        prev_xp0 = test_case.trajectory_dict[0][idx-1]
        prev_xp1 = test_case.trajectory_dict[1][idx-1]
        prev_acas0 = prev_xp0[4]
        prev_acas1 = prev_xp1[4]

        # Obtain RA and desired velocity for agent 0 
        ra0 = self.generate_acas(agent0_state, agent1_state, 0, prev_acas0)
        xp0[4] = ra0
        p0 = self.process_acas(ra0, agent0_state, test_idx, 0)
        xp0[1] = p0
        test_case.trajectory_dict[0][idx] = xp0

        # Obtain RA and desired velocity for agent 1
        ra1 = self.generate_acas(agent1_state, agent0_state, 0, prev_acas1)
        xp1[4] = ra1
        p1 = self.process_acas(ra1, agent1_state, test_idx, 1)
        xp1[1] = p1
        test_case.trajectory_dict[1][idx] = xp1
        test_case.simulation_time += 1
        return [xp0, xp1]

    def initialize_stack(self):
        for test in self.test_cases:
            xp0 = test.trajectory_dict[0][0]
            xp1 = test.trajectory_dict[1][0]
            agent0_state = xp0[0]
            agent1_state = xp1[0]
            ra0 = self.generate_acas(agent0_state, agent1_state, 0, RA.COC)
            xp0[4] = ra0
            p0 = self.process_acas(ra0, agent0_state, test.idx, 0)
            xp0[1] = p0
            test.trajectory_dict[0][0] = xp0
            ra1 = self.generate_acas(agent1_state, agent0_state, 0, RA.COC)
            xp1[4] = ra1
            p1 = self.process_acas(ra1, agent1_state, test.idx, 1)
            xp1[1] = p1
            test.trajectory_dict[1][0] = xp1
            self.test_stack.append(xp0)
            self.test_stack.append(xp1)

    def generate_key_2(self, state, desired_velocity):
        roll = state[6]
        pitch = state[7]
        yaw = state[8]

        vx = state[3]
        vy = state[4]
        vz = state[5]
        vbody = self.convert_pos_to_body([vx, vy, vz], roll, pitch, yaw)

        vdesired_x = desired_velocity[0]
        vdesired_y = desired_velocity[1]
        vdesired_z = desired_velocity[2]
        vdesired_body = self.convert_pos_to_body([vdesired_x, vdesired_y, vdesired_z], roll, pitch, yaw)

        vdesired_body_offset = [vdesired_body[0] - vbody[0], vdesired_body[1] - vbody[1], vdesired_body[2] - vbody[2]]
        # key_state = [roll, pitch] + vdesired_body_offset
        key_state = vdesired_body_offset
        state_round = []
        for j, val in enumerate(key_state):
            state_round.append(self.__round(val, self.resolution[j]))
        key = tuple(state_round)
        
        return key

    def __round(self, val, base):
        res = base * round(val/base)
        return res

    def in_cache(self, xp):
        state = xp[0]
        desired_velocity = xp[1]
        key = self.generate_key_2(state, desired_velocity)
                    
        if key in self.cache_dict:
            return True 
        else:
            return False

    def get_from_cache(self, elem):
        state = elem[0]
        desired_velocity = elem[1]

        # Get next state from cache
        next_state = self.get_next_state(state, desired_velocity)

        # Update next state in the physical trajectory for the test step
        xp = [next_state, (), elem[2], elem[3], RA.NA, 1]
        self.test_cases[elem[2]].trajectory_dict[elem[3]].append(xp)

    def get_next_state(self, state, desired_velocity):
        roll = state[6]
        pitch = state[7] 
        yaw = state[8]

        key = self.generate_key_2(state, desired_velocity)
                    
        new_state = copy.deepcopy(self.cache_dict[key])

        vx = state[3]
        vy = state[4]
        vz = state[5]
        v_body = self.convert_pos_to_body([vx, vy, vz], roll, pitch, yaw)

        # Handle offset for orientation
        new_state[6] += state[6]
        new_state[7] += state[7]
        new_state[8] += state[8]

        # Handle offset for velocity
        next_vx_body = new_state[3] + v_body[0]
        next_vy_body = new_state[4] + v_body[1]
        next_vz_body = new_state[5] + v_body[2]

        # Handle offset for position
        x_offset = new_state[0] + v_body[0]
        y_offset = new_state[1] + v_body[1]
        z_offset = new_state[2] + v_body[2]

        # Transform velocity and position offset back to world coordinate
        offset = self.convert_body_to_pos([x_offset, y_offset, z_offset], roll, pitch, yaw)
        next_v = self.convert_body_to_pos([next_vx_body, next_vy_body, next_vz_body], roll, pitch, yaw)

        # Compute new position
        new_state[0] = state[0] + offset[0]
        new_state[1] = state[1] + offset[1]
        new_state[2] = state[2] + offset[2]

        # Compute new velocity
        new_state[3] = next_v[0]
        new_state[4] = next_v[1]
        new_state[5] = next_v[2]

        return new_state

    def convert_body_to_pos(self, pos_body, roll, pitch, yaw):
        pos_body = np.array(pos_body)
        Rroll = [[1,0,0],[0, np.cos(roll), np.sin(roll)],[0, -np.sin(roll), np.cos(roll)]]
        Rpitch = [[np.cos(pitch),0 , -np.sin(pitch)],[0,1,0],[np.sin(pitch),0 ,np.cos(pitch)]]
        Ryaw = [[np.cos(yaw), np.sin(yaw), 0],[-np.sin(yaw), np.cos(yaw), 0],[0,0,1]]
        
        pos = np.linalg.inv(np.array(Rroll)@np.array(Rpitch)@np.array(Ryaw))@np.array(pos_body)
        pos = list(pos)
        return pos

    def convert_pos_to_body(self, pos, roll, pitch, yaw):
        pos = np.array(pos)
        Rroll = [[1,0,0],[0, np.cos(roll), np.sin(roll)],[0, -np.sin(roll), np.cos(roll)]]
        Rpitch = [[np.cos(pitch),0 , -np.sin(pitch)],[0,1,0],[np.sin(pitch),0 ,np.cos(pitch)]]
        Ryaw = [[np.cos(yaw), np.sin(yaw), 0],[-np.sin(yaw), np.cos(yaw), 0],[0,0,1]]
        
        pos_body = np.array(pos)
        pos_body = np.array(Ryaw)@np.array(pos)
        pos_body = np.array(Rroll)@np.array(Rpitch)@np.array(Ryaw)@np.array(pos)
        
        pos_body = list(pos_body)
        return pos_body

    def add_step(self, trajectories, desired_velocity):
        # desired_velocity = point[18:21]
        state = trajectories[0]
        key = self.generate_key_2(state, desired_velocity)

        next_state = copy.deepcopy(trajectories[1])
        x_offset = (next_state[0] - state[0] - state[3]) 
        y_offset = (next_state[1] - state[1] - state[4]) 
        z_offset = (next_state[2] - state[2] - state[5]) 

        roll = state[6]
        pitch = state[7]
        yaw = state[8]

        offset_body = self.convert_pos_to_body([x_offset, y_offset, z_offset], roll, pitch, yaw)    

        vx = state[3]
        vy = state[4]
        vz = state[5]
        v_body = self.convert_pos_to_body([vx, vy, vz], roll, pitch, yaw)

        next_vx = next_state[3]
        next_vy = next_state[4]
        next_vz = next_state[5]
        nextv_body = self.convert_pos_to_body([next_vx, next_vy, next_vz], roll, pitch, yaw)
        
        next_state[0] = offset_body[0]
        next_state[1] = offset_body[1]
        next_state[2] = offset_body[2]

        next_state[6] -= roll 
        next_state[7] -= pitch 
        next_state[8] -= yaw

        next_state[3] = nextv_body[0] - v_body[0]
        next_state[4] = nextv_body[1] - v_body[1]
        next_state[5] = nextv_body[2] - v_body[2]

        self.cache_dict[key] = next_state

    def perform_simulation(self, elem):
        # Compute the key that the current x,p pair is corresponding to
        state = elem[0] 
        desired_velocity = elem[1]
        roll = state[6]
        pitch = state[7]
        yaw = state[8]

        vx = state[3]
        vy = state[4]
        vz = state[5]
        vbody = self.convert_pos_to_body([vx, vy, vz], roll, pitch, yaw)

        vdesired_x = desired_velocity[0]
        vdesired_y = desired_velocity[1]
        vdesired_z = desired_velocity[2]
        vdesired_body = self.convert_pos_to_body([vdesired_x, vdesired_y, vdesired_z], roll, pitch, yaw)

        vdesired_body_offset = [vdesired_body[0] - vbody[0], vdesired_body[1] - vbody[1], vdesired_body[2] - vbody[2]]
        key = vdesired_body_offset
        
        # Use the key to compute the desired velocity for the agent at current state 
        agent_state = copy.deepcopy(self.agent.state)
        agent_vx = agent_state[3]
        agent_vy = agent_state[4]
        agent_vz = agent_state[5]
        agent_roll = agent_state[6]
        agent_pitch = agent_state[7]
        agent_yaw = agent_state[8]
        agent_v_body = self.convert_pos_to_body([agent_vx, agent_vy, agent_vz], agent_roll, agent_pitch, agent_yaw)
        agent_vdesired_body = [agent_v_body[0] + key[0], agent_v_body[1] + key[1], agent_v_body[2] + key[2]]
        agent_vdesired = self.convert_body_to_pos(agent_vdesired_body, agent_roll, agent_pitch, agent_yaw)

        # Use the desired velocity to compute the mode parameters and actually perform simulations
        plan = [
            agent_state[0], 
            agent_state[1], 
            agent_state[2], 
            agent_state[0]+ agent_vdesired[0],
            agent_state[1]+ agent_vdesired[1],
            agent_state[2]+ agent_vdesired[2]
        ]
        trajectory = self.agent.TC_Simulate(plan, agent_state[:6], 1)
        agent_new_state = copy.deepcopy(self.agent.state)

        # Add the simulated result to cache
        self.add_step([agent_state, agent_new_state], agent_vdesired)

        # Update actual trajectory in the test case
        next_state = self.get_next_state(state, desired_velocity)
        xp = [next_state, (), elem[2], elem[3], RA.NA, 0]
        self.test_cases[elem[2]].trajectory_dict[elem[3]].append(xp)

    def perform_simulation_direct(self, elem):
        desired_velocity = elem[1]
        state = elem[0] 
        plan = [
            state[0], 
            state[1], 
            state[2], 
            state[0]+ desired_velocity[0],
            state[1]+ desired_velocity[1],
            state[2]+ desired_velocity[2]
        ]
        trajectory = self.agent.TC_Simulate(plan, state[:6], 1)
        next_state = self.agent.state
        xp = [next_state, (), elem[2], elem[3], RA.NA, 0]
        self.test_cases[elem[2]].trajectory_dict[elem[3]].append(xp)

    def run_tests_transformed(self):
        self.initialize_stack()
        total_points = 0
        cached_points = 0
        simulate_points = 0
        while self.test_stack != []:
            total_points += 1
            elem = self.test_stack.pop()
            if self.in_cache(elem):
                # Get corresponding trajectory from cache
                cached_points += 1
                self.get_from_cache(elem)
            else:
                # Simulate to get corresponding trajectory
                simulate_points += 1
                self.perform_simulation(elem)
            
            test_idx = elem[2]
            agent_idx = elem[3]
            test_case = self.test_cases[test_idx]
            if test_case.match(agent_idx):
                new_xp_pair = self.get_new_xp_pair(test_idx, agent_idx)
                if test_case.simulation_time < test_case.timehorizon:
                    self.test_stack += new_xp_pair
        print(total_points, simulate_points, cached_points)
        return self.test_cases, total_points, simulate_points, cached_points

    def run_simulation(self):
        self.initialize_stack()
        total_points = 0
        cached_points = 0
        simulate_points = 0
        while self.test_stack != []:
            total_points += 1
            elem = self.test_stack.pop()
            if False:
                # Get corresponding trajectory from cache
                cached_points += 1
                self.get_from_cache(elem)
            else:
                # Simulate to get corresponding trajectory
                simulate_points += 1
                self.perform_simulation_direct(elem)
            
            test_idx = elem[2]
            agent_idx = elem[3]
            test_case = self.test_cases[test_idx]
            if test_case.match(agent_idx):
                new_xp_pair = self.get_new_xp_pair(test_idx, agent_idx)
                if test_case.simulation_time < test_case.timehorizon:
                    self.test_stack += new_xp_pair
        print(total_points, simulate_points, cached_points)
        return self.test_cases, total_points, simulate_points, cached_points


if __name__ == "__main__":
    # The test case is defined by a list 
    # [theta, vint, rho, h_diff, time2collision, timehorizon]
    # test_cases = [
    #     [np.pi/2, 1050, 43736, 0, 100, 150],
    #     [-np.pi/2, 750, 43736, 0, 100, 150]
    # ]

    test_cases = []
    # theta_list = [-np.pi*3/4, -np.pi*3/8, np.pi/3]
    # vint_list = [600, 750, 900, 1050]
    # rho_list = [10000, 43736, 87472, ]

    theta_list = [-np.pi*3/4, -np.pi/2, -np.pi*3/8, -np.pi/4, 0, np.pi/4, np.pi/3, np.pi*3/4, np.pi]
    vint_list = [60, 150, 300, 450, 600, 750, 900, 1050, 1145]
    rho_list = [10000, 43736, 87472, 120000]

    for theta in theta_list:
        for vint in vint_list:
            for rho in rho_list:
                test_cases.append([theta, vint, rho, 0, 100, 150])


    test = NewAlgorithm(test_cases)
    res, total_points, simulate_points, cached_points = test.run_tests_transformed()
    
    test2 = NewAlgorithm(test_cases)
    res2, _, _, _ = test2.run_simulation()

    for j in range(len(res)):
        x0 = []
        y0 = []
        x1 = []
        y1 = []
        
        x0_simulate = []
        y0_simulate = []
        x1_simulate = []
        y1_simulate = []
        
        x0_cache = []
        y0_cache = []
        x1_cache = []
        y1_cache = []

        x0_2 = []
        y0_2 = []
        x1_2 = []
        y1_2 = []

        test_result = res[j]
        # print(f"trajectory 0 length {len(test_result.trajectory_dict[0])}")
        # print(f"trajectory 1 length {len(test_result.trajectory_dict[1])}")
        for i in range(len(test_result.trajectory_dict[0])):
            # Plot agent 0
            x = test_result.trajectory_dict[0][i][0][0]
            y = test_result.trajectory_dict[0][i][0][1]
            # plt.plot(x, y, 'r.')
            x0.append(x)
            y0.append(y)
            if test_result.trajectory_dict[0][i][-1] == 1:
                x0_cache.append(x)
                y0_cache.append(y)
                x0_simulate.append(None)
                y0_simulate.append(None)
            else:
                x0_cache.append(None)
                y0_cache.append(None)
                x0_simulate.append(x)
                y0_simulate.append(y)
            
            # PLot agent 1
            x = test_result.trajectory_dict[1][i][0][0]
            y = test_result.trajectory_dict[1][i][0][1]
            # plt.plot(x, y, 'b.')
            x1.append(x)
            y1.append(y)
            if test_result.trajectory_dict[1][i][-1] == 1:
                x1_cache.append(x)
                y1_cache.append(y)
                x1_simulate.append(None)
                y1_simulate.append(None)
            else:
                x1_cache.append(None)
                y1_cache.append(None)
                x1_simulate.append(x)
                y1_simulate.append(y)
            
        test_result = res2[j]
        print(f"trajectory 0 length {len(test_result.trajectory_dict[0])}")
        print(f"trajectory 1 length {len(test_result.trajectory_dict[1])}")
        for i in range(len(test_result.trajectory_dict[0])):
            # Plot agent 0
            x = test_result.trajectory_dict[0][i][0][0]
            y = test_result.trajectory_dict[0][i][0][1]
            # plt.plot(x, y, 'r.')
            x0_2.append(x)
            y0_2.append(y)

            # PLot agent 1
            x = test_result.trajectory_dict[1][i][0][0]
            y = test_result.trajectory_dict[1][i][0][1]
            # plt.plot(x, y, 'b.')
            x1_2.append(x)
            y1_2.append(y)

        plt.figure(1)
        plt.plot(x0, y0, 'b')
        plt.plot(x1, y1, 'b')
        plt.plot(x0_cache, y0_cache, 'r.')
        plt.plot(x1_cache, y1_cache, 'r.')
        plt.plot(x0_simulate, y0_simulate, 'g.')
        plt.plot(x1_simulate, y1_simulate, 'g.')

        plt.plot(x0_2, y0_2, 'y')
        plt.plot(x1_2, y1_2, 'y')
        # plt.plot(x0, y0, 'r.')
        # plt.plot(x1, y1, 'r.')
        agent0_array = np.array([x0, y0])
        agent1_array = np.array([x1, y1])
        tmp = np.linalg.norm(agent0_array - agent1_array, axis = 0)
        # plt.figure(2)
        # plt.plot(tmp)
        # plt.show()
        plt.savefig(f'./res/test{j}.png')
        plt.clf()
    print(f"total points: {total_points}, simulated points {simulate_points}, cached points {cached_points}, hit rate {cached_points/total_points}, resolution {test.resolution} ")

    simulate_agent_trajectory = np.array(test.agent.trajectory)
    plt.plot(simulate_agent_trajectory[:,0], simulate_agent_trajectory[:,1], 'b')
    plt.plot(simulate_agent_trajectory[:,0], simulate_agent_trajectory[:,1], 'g.')
    plt.savefig(f'./res/simulate_agent_trajectory.png')
    plt.show()
