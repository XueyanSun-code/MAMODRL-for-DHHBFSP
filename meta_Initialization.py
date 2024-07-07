import numpy as np
import random
import copy
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')


class Initializations():
    def __init__(self, numjob, numstage, nummachine, process_time,time_f,energy_fac):
        self.numjob = numjob
        self.numstage = numstage
        self.nummachine = nummachine#heterogeneous ,这里时每个车间的各个阶段的机器数不同
        self.process_time= process_time
        self.time_f = time_f#单车间的每个阶段

        self.energy_fac = energy_fac

        self.blocking_f = 0.3
        self.idle_f = 0.1
        #

    def population_initial(self, order, polulation_size,f):
        # order_set = []
        population = []
        makespan_pop = []
        new_order,  makespan = self.NEH_LPT(order,f)
        population.append(new_order)
        makespan_pop.append(makespan)

        # ##NEH_SPT
        new_order,  makespan = self.NEH_SPT(order,f)
        population.append(new_order)
        makespan_pop.append(makespan)


        # ##SPT_first（前面的阶段）idle time
        new_order,  makespan =self.NEH_SPT_first(order,f)
        population.append(new_order)
        makespan_pop.append(makespan)
        # print('orderNEH-first: ' + str(new_order))
        # print('makespan ' + str(makespan))
        # ##SPT_last（后面的阶段）blocking time
        new_order,  makespan = self.NEH_SPT_last(order,f)
        population.append(new_order)
        makespan_pop.append(makespan)
        # print('orderNEH-last: ' + str(new_order))
        # print('makespan ' + str(makespan))
        # ##LPT_first（前面的阶段）idle time
        new_order,  makespan =self.NEH_LPT_first(order,f)
        population.append(new_order)
        makespan_pop.append(makespan)
        # print('orderNEH-first: ' + str(new_order))
        # print('makespan ' + str(makespan))
        # ##SPT_last（后面的阶段）blocking time
        new_order,  makespan = self.NEH_LPT_last(order,f)
        population.append(new_order)
        makespan_pop.append(makespan)
        # print('orderNEH-last: ' + str(new_order))
        # print('makespan ' + str(makespan))


           ##随机产生个体
        while len(population) < polulation_size:
            order_ = copy.copy(order)
            random.shuffle(order_)

            population.append(order_)
            makespan_pop.append(self.fitness_bi(order_,f))
            # print(str(order_list))
        return population, makespan_pop

    def Not_exist_same_one(self, order_list, population):
        flag = True
        for x, pop in enumerate(population):
            for i, item in enumerate(pop):
                if len(item) == len(order_list[i]):
                    for j, job in enumerate(item):
                        if job != order_list[i][j]:
                            break
                        if i == len(pop) - 1 and j == len(item) - 1:
                            flag = False
                            return flag

                    else:
                        continue
                    break
        return flag



    def NEH2_LPT_assignment(self, order):
        sum_t = np.sum(self.process_time, axis=1)
        order = sorted(order, reverse=True, key=lambda k: sum_t[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2_SPT_assignment(self, order):
        sum_t = np.sum(self.process_time, axis=1)
        order = sorted(order, key=lambda k: sum_t[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2_SPT_first_assignment(self, order):
        sum_t_f = np.sum(self.process_time[:, :self.numstage - 1], axis=1)
        order = sorted(order, key=lambda k: sum_t_f[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2_SPT_last_assignment(self, order):
        sum_t_l = np.sum(self.process_time[:, 1:], axis=1)
        order = sorted(order, key=lambda k: sum_t_l[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2_LPT_first_assignment(self, order):
        sum_t_f = np.sum(self.process_time[:, :self.numstage - 1], axis=1)
        order = sorted(order, reverse=True, key=lambda k: sum_t_f[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2_LPT_last_assignment(self, order):
        sum_t_l = np.sum(self.process_time[:, 1:], axis=1)
        order = sorted(order, reverse=True, key=lambda k: sum_t_l[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2L(self, order):
        sum_t = np.sum(self.process_time, axis=1)
        ave_E = sum_t / self.numstage
        # 计算方差,参数alpha 是0.6，根据文献2021chen
        sum_D = np.zeros(len(order), dtype=int)
        sum_error = np.zeros(len(order), dtype=int)
        for i, item in enumerate(order):
            for j in range(self.numstage):
                sum_error[item - 1] = sum_error[item - 1] + (self.process_time[item - 1, j] - ave_E[item - 1]) ** 2
            sum_D[item - 1] = sum_error[item - 1] ** 0.5
        alpha = 0.6
        ED = ave_E * alpha + sum_D * (1 - alpha)
        order = sorted(order, reverse=True, key=lambda k: ED[k - 1])
        new_order, makespan = self.NEH2_step2(order)
        return new_order, makespan

    def NEH2D(self, order):
        sum_t = np.sum(self.process_time, axis=1)  # 按照process_time的顺序，0，1，2，3......
        ave_E = sum_t / self.numstage
        # 计算方差,参数alpha 是0.6，根据文献2021chen
        sum_STD = np.zeros(len(order), dtype=int)
        sum_error = np.zeros(len(order), dtype=int)
        for i, item in enumerate(order):
            for j in range(self.numstage):
                sum_error[item - 1] = sum_error[item - 1] + (self.process_time[item - 1, j] - ave_E[item - 1]) ** 2
            sum_STD[item - 1] = (sum_error[item - 1] / (self.numstage - 1)) ** 0.5
        STD = ave_E + sum_STD
        order = sorted(order, reverse=True, key=lambda k: STD[k - 1])
        new_order, makespan = self.NEH2_step2(order)
        return new_order, makespan

    def NEH2E(self, order):
        sum_t = np.sum(self.process_time, axis=1)
        ave_E = sum_t / self.numstage
        alpha = 0.6
        WD = np.zeros(len(order), dtype=int)
        for i, item in enumerate(order):
            for j in range(self.numstage - 1):
                WD[item - 1] = WD[item - 1] + (self.numstage - j) / self.numstage * max(0,
                                                                                        self.process_time[item - 1, j] -
                                                                                        self.process_time[
                                                                                            item - 1, j + 1])
        EWD = ave_E * alpha + WD * (1 - alpha)
        order = sorted(order, reverse=True, key=lambda k: EWD[k - 1])
        new_order, makespan = self.NEH2_step2(order)
        return new_order, makespan

    def NEH2E_ex(self, order):
        sum_t = np.sum(self.process_time, axis=1)
        ave_E = sum_t / self.numstage
        alpha = 0.6
        WD = np.zeros(len(order), dtype=int)
        for i, item in enumerate(order):
            for j in range(self.numstage - 1):
                WD[item - 1] = WD[item - 1] + (self.numstage - j) / self.numstage * max(0,
                                                                                        self.process_time[item - 1, j] -
                                                                                        self.process_time[
                                                                                            item - 1, j + 1])
        EWD = ave_E * alpha + WD * (1 - alpha)
        order = sorted(order, reverse=True, key=lambda k: EWD[k - 1])
        ##NEH step 2
        order_list = [[] for i in range(self.numfactory)]
        order_list[0].append(order[0])
        left_order = order[1:]
        makespan_list = [0 for i in range(self.numfactory)]
        makespan_list[0] = np.sum(self.process_time[order[0] - 1, :])
        for j in range(len(left_order)):
            # 在所以车间的所有位置选出最好的位置插入
            order_list, makespan_list, best_fac_, id_ = self.job_insertion3(order_list, left_order[j], makespan_list)
            ##完成NEH2E-ex，前面的工件选择最佳位置插入后，若order大于两个工件，则进行插入工件的前一个或后一个随机再插入到所有车间
            new_order = order_list[best_fac_]
            if len(new_order) > 2:
                if id_ == 0:  ##第二个工件被删除
                    left = new_order[id_ + 1]
                    del new_order[id_ + 1]

                elif id_ == len(new_order) - 1:  ##倒数第二个被删除
                    left = new_order[id_ - 1]
                    del new_order[id_ - 1]
                else:  ##随机选择前后一个删除
                    if random.random() > 0.5:
                        left = new_order[id_ + 1]
                        del new_order[id_ + 1]
                    else:
                        left = new_order[id_ - 1]
                        del new_order[id_ - 1]
                makespan_list[best_fac_] = self.fitness(new_order)
                ##进行重新插入操作
                order_list, makespan_list = self.job_insertion2(order_list, left, makespan_list)

        return order_list, makespan_list

    def NEH2EE(self, order):
        ##根据NEH2e生成初始解
        sum_t = np.sum(self.process_time, axis=1)
        p2 = sum_t / self.numstage
        ##计算两次插入的顺序
        p1 = copy.copy(p2)
        double_order = np.zeros(2 * len(order), dtype=int)
        PV = np.zeros(2 * len(order))
        for i, item in enumerate(order):
            for j in range(self.numstage - 1):
                p1[item - 1] = p1[item - 1] + 1 / self.numstage * abs(
                    self.process_time[item - 1, j] - self.process_time[item - 1, j + 1])
            double_order[i * 2:i * 2 + 2] = [i + 1, i + 1]
            PV[i * 2:i * 2 + 2] = [p1[item - 1], p2[item - 1]]
        ##先排序再进行插入操作
        order_temp = range(2 * len(order))
        order_temp = sorted(order_temp, key=lambda k: PV[k])
        double_order_new = []
        for i, item in enumerate(order_temp):
            double_order_new.append(double_order[item])
        # print(str(double_order_new))
        order_list = [[] for i in range(self.numfactory)]
        makespan_list = [0 for i in range(self.numfactory)]
        ##先删除重复的工件，再插入
        ##判断当前的list是否有此工件
        for i, item in enumerate(double_order_new):
            for order1 in order_list:
                if item in order1:
                    del order1[order1.index(item)]
                    break

            order_list, makespan_list = self.job_insertion2(order_list, item, makespan_list)
        return order_list, makespan_list



    def NEH_LPT(self, order,f):
        sum_t = np.sum(self.process_time, axis=1)
        order = sorted(order, reverse=True, key=lambda k: sum_t[k - 1])
        # makespan = self.fitness(order)
        # print('orderLPT: ' + str(order))
        # print('makespan ' + str(makespan))
        new_order,  makespan = self.NEH_step2(order,f)
        return new_order,  makespan

    def NEH_SPT(self, order,f):
        sum_t = np.sum(self.process_time, axis=1)
        order = sorted(order, key=lambda k: sum_t[k - 1])
        # makespan = self.fitness(order)
        # print('orderSPT: ' + str(order))
        # print('makespan ' + str(makespan))
        new_order, makespan = self.NEH_step2(order,f)
        return new_order,  makespan

    def NEH_SPT_first(self, order,f):
        sum_t_f = np.sum(self.process_time[:,:self.numstage - 1], axis=1)
        order = sorted(order, key=lambda k: sum_t_f[k - 1])
        # makespan = self.fitness(order)
        # print('orderfirst: ' + str(order))
        # print('makespan ' + str(makespan))
        new_order,  makespan = self.NEH_step2(order,f)
        return new_order,  makespan

    def NEH_SPT_last(self, order,f):
        sum_t_l = np.sum(self.process_time[:, 1:], axis=1)
        order = sorted(order, key=lambda k: sum_t_l[k - 1])
        # makespan = self.fitness(order)
        # print('orderlast: ' + str(order))
        # print('makespan ' + str(makespan))
        new_order,  makespan = self.NEH_step2(order,f)
        return new_order,  makespan

    def NEH_LPT_first(self, order,f):
        sum_t_f = np.sum(self.process_time[:,:self.numstage - 1], axis=1)
        order = sorted(order, reverse=True,key=lambda k: sum_t_f[k - 1])
        # makespan = self.fitness(order)
        # print('orderfirst: ' + str(order))
        # print('makespan ' + str(makespan))
        new_order,  makespan = self.NEH_step2(order,f)
        return new_order,  makespan

    def NEH_LPT_last(self, order,f):
        sum_t_l = np.sum(self.process_time[:, 1:], axis=1)
        order = sorted(order,reverse=True, key=lambda k: sum_t_l[k - 1])
        # makespan = self.fitness(order)
        # print('orderlast: ' + str(order))
        # print('makespan ' + str(makespan))
        new_order,  makespan = self.NEH_step2(order,f)
        return new_order,  makespan


    def NEH_step2(self, order, f):
        ##NEH step 2
        new_order = [order[0]]
        makespan = self.fitness_bi(new_order,f)
        left_order = order[1:]
        for j in range(len(left_order)):
            new_order,  makespan = self.job_insertion(new_order, left_order[j], f)
        return new_order,  makespan
    #
    # def job_insertion2(self, order_list, insert_job, makespan):
    #     order_store = []
    #     makespan_max = []  # 包含每个组合list所有车间中max的makespan
    #     makespan_all = []  # 包含每个车间的makespan
    #     for f in range(self.numfactory):
    #         makespan_temp = copy.copy(makespan)
    #         order_f = copy.copy(order_list)
    #         for i in range(len(order_list[f]) + 1):
    #             if i == 0:
    #                 order_temp = [insert_job] + order_list[f]
    #
    #             elif i == len(order_list[f]):
    #                 order_temp = order_list[f] + [insert_job]
    #             else:
    #                 order_temp = order_list[f][:i] + [insert_job] + order_list[f][i:]
    #             ##计算makespan
    #             makespan_temp[f] = self.fitness(order_temp)
    #             ms = max(makespan_temp)
    #             makespan_max.append(ms)
    #             makespan_all.append(copy.copy(makespan_temp))
    #             order_f[f] = order_temp
    #             ##存储order_temp
    #             order_store.append(copy.deepcopy(order_f))
    #
    #     ##选择makespan最大的new_order
    #     id_ = np.argmin(makespan_max)
    #     order_list = order_store[id_]
    #     makespan = makespan_all[id_]
    #     return order_list, makespan

    def job_insertion(self, order, insert_job,f):
        makespan_temp = []
        order_store = []
        for i in range(len(order) + 1):
            if i == 0:
                order_temp = [insert_job] + order

            elif i == len(order):
                order_temp = order + [insert_job]
            else:
                order_temp = order[:i] + [insert_job] + order[i:]
            ##计算makespan
            makespan_temp.append(self.fitness_bi(order_temp,f))
            ##存储order_temp
            order_store.append(order_temp)

        ##选择makespan最大的new_order
        PE = []
        for i in range(len(makespan_temp)):
            temp = self.caculate_ave_PE_NN(makespan_temp[i][0], makespan_temp[i][1])
            PE.append(temp)
        # order = list(range(len(makespan_max)))
        # order_ = sorted(order, reverse=True, key=lambda k: PE[k - 1])
        id_ = np.argmin(PE)
        order = order_store[id_]
        objs = makespan_temp[id_]
        return order,  objs

    def caculate_ave_PE_NN(self,ave_process_time,ave_energy_con):
        '''随机选择50组权重，找到使得Q最大的权重或者是使用平均值进行计算loss'''
        set_num = 20
        weight_sum = np.ones(set_num)
        weight1 = np.random.rand(set_num)
        weight2 = weight_sum - weight1
        PE = 0

        for i in range(set_num):
            PE += weight1[i] * ave_process_time + weight2[i] * ave_energy_con
        return PE

    def fitness_bi(self, order, f):
        ##这里包含了两个目标函数的值 makespan 和 EC
        fitness = []
        EC = 0
        start_end_block_time = np.zeros((self.numjob, self.numstage * 3), dtype=int)
        machine_complete_time = []
        machine_job_list = []
        for i in range(self.numstage):
            time = [0 for j in range(self.nummachine[i])]
            job_list = [[] for j in range(self.nummachine[i])]
            machine_complete_time.append(time)
            machine_job_list.append(job_list)

        for i, item in enumerate(order):
            for j in range(self.numstage):
                id_ = np.argmin(machine_complete_time[j])
                MCT = copy.copy(machine_complete_time[j])
                if j == 0:
                    start_end_block_time[item - 1, j * 3] = machine_complete_time[j][id_]
                    start_end_block_time[item - 1, j * 3 + 1] = start_end_block_time[item - 1, j * 3] + \
                                                                self.process_time[item - 1][j] *self.time_f[j]
                    machine_complete_time[j][id_] = max(start_end_block_time[item - 1, j * 3 + 1],
                                                        min(machine_complete_time[j + 1]))
                    start_end_block_time[item - 1, j * 3 + 2] = machine_complete_time[j][id_] - start_end_block_time[
                        item - 1, j * 3 + 1]
                    start_end_block_time[item - 1, j * 3 + 1] = machine_complete_time[j][id_]

                elif j == self.numstage - 1:
                    start_end_block_time[item - 1, j * 3] = start_end_block_time[item - 1, (j - 1) * 3 + 1]
                    start_end_block_time[item - 1, j * 3 + 1] = \
                        start_end_block_time[item - 1, j * 3] + self.process_time[item - 1][j]*self.time_f[j]
                    machine_complete_time[j][id_] = start_end_block_time[item - 1, j * 3 + 1]
                else:

                    start_end_block_time[item - 1, j * 3] = start_end_block_time[item - 1, (j - 1) * 3 + 1]
                    start_end_block_time[item - 1, j * 3 + 1] = \
                        start_end_block_time[item - 1, j * 3] + self.process_time[item - 1][j]*self.time_f[j]
                    machine_complete_time[j][id_] = max(start_end_block_time[item - 1, j * 3 + 1],
                                                        min(machine_complete_time[j + 1]))
                    start_end_block_time[item - 1, j * 3 + 2] = machine_complete_time[j][id_] - start_end_block_time[
                        item - 1, j * 3 + 1]
                    start_end_block_time[item - 1, j * 3 + 1] = machine_complete_time[j][id_]
                ##计算EC
                print('', )
                Ep = self.process_time[item - 1][j] * self.time_f[j] *self.energy_fac[j]
                Eb = start_end_block_time[item - 1, j * 3 + 2] * self.energy_fac[j] * self.blocking_f
                Ei = (start_end_block_time[item - 1, j * 3] - MCT[id_])* self.energy_fac[j] * self.idle_f##现在的开始时间减去原来本机器的完成时间
                EC = EC + Ep + Eb +Ei


        makespan = max(start_end_block_time[:, -2])
        fitness.append(makespan)
        fitness.append(EC)

        return fitness

    def makespan_chrome(self, order_list):
        makespan_list = []
        for temp in order_list:
            if temp:
                makespan_list.append(self.fitness(temp))
            else:
                makespan_list.append(0)

        return makespan_list
