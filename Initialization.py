import numpy as np
import random
import copy
import matplotlib.pyplot as plt



plt.style.use('seaborn-white')

class Initializations():
    def __init__(self,numfactory, numjob, numstage, nummachine, process_time):
        self.numfactory = numfactory
        self.numjob = numjob
        self.numstage = numstage
        self.nummachine = nummachine
        self.process_time= process_time
        #

    def population_initial(self, order,polulation_size):
        order_set = []
        population = []
        makespan_pop = []
        # new_order, id_, makespan = self.NEH_LPT(order)
        # order_set.append(new_order)
        # #print('orderNEH-LPT: ' + str(new_order))
        # #print('makespan ' + str(makespan))
        #
        # # ##NEH_SPT
        # new_order, id_, makespan = self.NEH_SPT(order)
        # order_set.append(new_order)
        # # print('orderNEH-SPT: ' + str(new_order))
        # # print('makespan ' + str(makespan))
        #
        # # ##SPT_first（前面的阶段）idle time
        # new_order, id_, makespan =self.NEH_first(order)
        # order_set.append(new_order)
        # # print('orderNEH-first: ' + str(new_order))
        # # print('makespan ' + str(makespan))
        # # ##SPT_last（后面的阶段）blocking time
        # new_order, id_, makespan = self.NEH_last(order)
        # order_set.append(new_order)
        #
        # # print('orderNEH-last: ' + str(new_order))
        # # print('makespan ' + str(makespan))
        #
        # for item in order_set:
        #     ##顺序分配
        #     order_list = self.sequence_assignment(item)
        #     ##判断是否有重复
        #     if self.Not_exist_same_one(order_list,population):
        #         population.append(order_list)
        #         makespan_list = self.makespan_chrome(order_list)
        #         makespan_pop.append(makespan_list)
        #     #print(str(order_list))
        #     ##按照现有makespan最小进行分配
        #     order_list, makespan_list = self.real_makespan_assignment(item)
        #     ##判断是否有重复
        #     if self.Not_exist_same_one(order_list,population):
        #         population.append(order_list)
        #         makespan_pop.append(makespan_list)
        #     # print(str(order_list))
        #     # print(str(makespan_list))
        #     ##按照插入后的估计值进行分配
        #     order_list, makespan_list = self.estimated_makespan_assignment(item)
        #     if self.Not_exist_same_one(order_list,population):
        #         population.append(order_list)
        #         makespan_pop.append(makespan_list)
        #     # print(str(order_list))
            # print(str(makespan_list))

        # ------#在NEH插入同时考虑工厂分配
        ##NEH2_LPT
        order_list, makespan_list = self.NEH2_LPT_assignment(order)
        ##判断是否有重复
        if self.Not_exist_same_one(order_list,population):
            population.append(order_list)
            makespan_pop.append(makespan_list)
        # print(str(order_list))
        # print(str(makespan_list))
        # order_list, makespan_list = self.NEH2_SPT_assignment(order)
        # ##判断是否有重复
        # if self.Not_exist_same_one(order_list,population):
        #     population.append(order_list)
        #     makespan_pop.append(makespan_list)
        # # print(str(order_list))
        # # print(str(makespan_list))
        # order_list, makespan_list = self.NEH2_SPT_first_assignment(order)
        # ##判断是否有重复
        # if self.Not_exist_same_one(order_list,population):
        #     population.append(order_list)
        #     makespan_pop.append(makespan_list)
        # # print(str(order_list))
        # # print(str(makespan_list))
        # order_list, makespan_list = self.NEH2_SPT_last_assignment(order)
        # ##判断是否有重复
        # if self.Not_exist_same_one(order_list,population):
        #     population.append(order_list)
        #     makespan_pop.append(makespan_list)
        # print(str(order_list))
        # print(str(makespan_list))
        order_list, makespan_list = self.NEH2_LPT_first_assignment(order)
        ##判断是否有重复
        if self.Not_exist_same_one(order_list,population):
            population.append(order_list)
            makespan_pop.append(makespan_list)
        # print(str(order_list))
        # print(str(makespan_list))
        order_list, makespan_list = self.NEH2_LPT_last_assignment(order)
        ##判断是否有重复
        if self.Not_exist_same_one(order_list,population):
            population.append(order_list)
            makespan_pop.append(makespan_list)
        # print(str(order_list))
        # print(str(makespan_list))
        order_list, makespan_list = self.NEH2E_ex(order)
        ##判断是否有重复
        if self.Not_exist_same_one(order_list, population):
            population.append(order_list)
            makespan_pop.append(makespan_list)

        order_list, makespan_list = self.NEH2D(order)
        ##判断是否有重复
        if self.Not_exist_same_one(order_list, population):
            population.append(order_list)
            makespan_pop.append(makespan_list)
        # print(str(order_list))
        # print(str(makespan_list))
        # order_list, makespan_list = self.NEH2EE(order)
        # ##判断是否有重复
        # if self.Not_exist_same_one(order_list, population):
        #     population.append(order_list)
        #     makespan_pop.append(makespan_list)
        # # print(str(order_list))
        # # print(str(makespan_list))
        # order_list, makespan_list = self.DPW(order)
        # ##判断是否有重复
        # if self.Not_exist_same_one(order_list, population):
        #     population.append(order_list)
        #     makespan_pop.append(makespan_list)
        # # print(str(order_list))
        # print(str(makespan_list))
        ##随机产生个体
        while len(population)< polulation_size:
            random.shuffle(order)
            order_list = self.sequence_assignment(order)
            ##判断是否有重复
            if self.Not_exist_same_one(order_list,population):
                population.append(order_list)
                makespan_pop.append(self.makespan_chrome(order_list))
            #print(str(order_list))
        return population, makespan_pop

    def Not_exist_same_one(self,order_list,population):
        flag = True
        for x, pop in enumerate(population):
            for i,item in enumerate(pop):
                if len(item) == len(order_list[i]):
                    for j, job in enumerate(item):
                        if job != order_list[i][j]:
                            break
                        if i == len(pop)-1 and j == len(item)-1:
                            flag = False
                            return flag

                    else:
                        continue
                    break
        return flag


    def sequence_assignment(self,order):
        order_list = [[] for i in range(self.numfactory)]
        for i, item in enumerate(order):
            temp = i % self.numfactory
            order_list[temp].append(item)
        return order_list

    def real_makespan_assignment(self,order):
        order_list = [[] for i in range(self.numfactory)]
        makespan_list = [0 for i in range(self.numfactory)]
        for i, item in enumerate(order):
            temp = np.argmin(makespan_list)
            order_list[temp].append(item)
            ##计算插入工件之后的makespan
            makespan_list[temp] = self.fitness(order_list[temp])
        return order_list, makespan_list

    def estimated_makespan_assignment(self,order):
        order_list = [[] for i in range(self.numfactory)]
        makespan_list = [0 for i in range(self.numfactory)]
        for i, item in enumerate(order):
            makespan_temp = [0 for i in range(self.numfactory)]
            for j in range(self.numfactory):
                order_temp = copy.deepcopy(order_list[j])
                order_temp.append(item)
                makespan_temp[j] = self.fitness(order_temp)
            temp = np.argmin(makespan_temp)
            order_list[temp].append(item)
            ##计算插入工件之后的makespan
            makespan_list[temp] = min(makespan_temp)
        return order_list, makespan_list

    def NEH2_LPT_assignment(self,order):
        sum_t = np.sum(self.process_time, axis=1)
        order = sorted(order, reverse=True, key=lambda k: sum_t[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2_SPT_assignment(self,order):
        sum_t = np.sum(self.process_time, axis=1)
        order = sorted(order, key=lambda k: sum_t[k - 1])

        order_list, makespan = self.NEH2_step2( order)
        return order_list, makespan

    def NEH2_SPT_first_assignment(self,order):
        sum_t_f = np.sum(self.process_time[:, :self.numstage - 1], axis=1)
        order = sorted(order, key=lambda k: sum_t_f[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2_SPT_last_assignment(self,order):
        sum_t_l = np.sum(self.process_time[:, 1:], axis=1)
        order = sorted(order, key=lambda k: sum_t_l[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2_LPT_first_assignment(self,order):
        sum_t_f = np.sum(self.process_time[:, :self.numstage - 1], axis=1)
        order = sorted(order, reverse=True, key=lambda k: sum_t_f[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2_LPT_last_assignment(self,order):
        sum_t_l = np.sum(self.process_time[:, 1:], axis=1)
        order = sorted(order, reverse=True, key=lambda k: sum_t_l[k - 1])

        order_list, makespan = self.NEH2_step2(order)
        return order_list, makespan

    def NEH2L(self,order):
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
        return new_order,  makespan

    def NEH2D(self, order):
        sum_t = np.sum(self.process_time, axis=1)#按照process_time的顺序，0，1，2，3......
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
                WD[item - 1] = WD[item - 1] + (self.numstage - j) / self.numstage * max(0, self.process_time[item - 1, j] -
                                                                self.process_time[item - 1, j + 1])
        EWD = ave_E * alpha + WD * (1 - alpha)
        order = sorted(order, reverse=True, key=lambda k: EWD[k - 1])
        new_order, makespan = self.NEH2_step2(order)
        return new_order,  makespan

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
            #在所以车间的所有位置选出最好的位置插入
            order_list, makespan_list,best_fac_,id_ = self.job_insertion3(order_list, left_order[j], makespan_list)
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
                order_list,  makespan_list = self.job_insertion2(order_list, left,makespan_list)

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
                p1[item - 1] = p1[item - 1] + 1 / self.numstage * abs(self.process_time[item - 1, j] - self.process_time[item - 1, j + 1])
            double_order[i * 2:i * 2 + 2] = [i + 1, i + 1]
            PV[i * 2:i * 2 + 2] = [p1[item - 1], p2[item - 1]]
        ##先排序再进行插入操作
        order_temp = range(2 * len(order))
        order_temp = sorted(order_temp, key=lambda k: PV[k])
        double_order_new = []
        for i, item in enumerate(order_temp):
            double_order_new.append(double_order[item])
        # print(str(double_order_new))
        order_list= [[] for i in range(self.numfactory)]
        makespan_list = [0 for i in range(self.numfactory)]
        ##先删除重复的工件，再插入
        ##判断当前的list是否有此工件
        for i, item in enumerate(double_order_new):
            for order1 in order_list:
                if item in order1:
                    del order1[order1.index(item)]
                    break

            order_list,  makespan_list = self.job_insertion2(order_list, item,makespan_list)
        return order_list,  makespan_list

    def DPW(self, order1):
        ##工序开始时间，结束时间，blocking时间
        end_time = np.zeros((self.numjob, self.numstage), dtype=int)
        all_machine_complete_time = []
        for f in range(self.numfactory):
            internal_machine_complete_time = []
            for i in range(self.numstage):
                time = [0 for j in range(self.nummachine[i])]
                internal_machine_complete_time.append(time)
            all_machine_complete_time.append(internal_machine_complete_time)
        ##这里的order充当U未安排的工件的集合
        ##计算每个工件的fj0 = (n-0-2)dalta+gamma, 初始的dalta =0

        order = copy.copy(order1)
        order_list = [[] for i in range(self.numfactory)]
        makespan_list = [0 for i in range(self.numfactory)]
        for k in range(self.numjob - 1):  ##最后一个工件自然加到最后
            fjk = []
            ##先选出makespan最小的factory
            fac = np.argmin(makespan_list)
            for i, item in enumerate(order):
                job_end_time = np.zeros(self.numstage, dtype=int)
                temp_machine_release_time = copy.deepcopy(all_machine_complete_time[fac])
                temp_order = copy.copy(order)
                del temp_order[i]
                dalta = 0
                ##更新每个工件的结束时间
                for j in range(self.numstage):
                    id_ = np.argmin(temp_machine_release_time[j])
                    if j == 0:
                        job_end_time[j] = temp_machine_release_time[j][id_]+ self.process_time[item - 1][j]
                        temp_machine_release_time[j][id_] = max(job_end_time[j], min(temp_machine_release_time[j+1]))
                        job_end_time[j] = temp_machine_release_time[j][id_]

                    elif j == self.numstage - 1:
                        temp_machine_release_time[j][id_] = job_end_time[j-1] + self.process_time[item - 1, j]
                    else:
                        temp_machine_release_time[j][id_] = max(max(job_end_time[j-1], temp_machine_release_time[j][id_]) + self.process_time[item - 1, j], min(temp_machine_release_time[j+1]))
                        job_end_time[j] = temp_machine_release_time[j][id_]

                ##计算dalta
                    dalta = dalta + self.numstage / (j+1 + len(order_list[fac]) * (self.numstage - j) / (self.numjob - 2)) * (
                            temp_machine_release_time[j][id_] - all_machine_complete_time[fac][j][id_] - self.process_time[item - 1][j])
                ##计算虚拟工件的加工时间
                virtual_job_time = np.zeros(self.numstage, dtype=int)
                gamma = 0
                for j in range(self.numstage):
                    for a, aitem in enumerate(temp_order):
                        virtual_job_time[j] = virtual_job_time[j] + self.process_time[aitem - 1][j]
                virtual_job_time = np.trunc(virtual_job_time / len(temp_order))
                ##计算加上虚拟工件后的releasetime
                virtual_release_time = np.zeros(self.numstage, dtype=int)
                virtual_end_time = np.zeros(self.numstage, dtype=int)
                for j in range(self.numstage):
                    id_ = np.argmin(temp_machine_release_time[j])
                    if j == 0:
                        virtual_release_time[j] = max(temp_machine_release_time[j][id_] + virtual_job_time[j],
                                                      min(temp_machine_release_time[j+1]))
                        virtual_end_time[j] = virtual_release_time[j]
                    elif j == self.numstage - 1:
                        virtual_release_time[j] = max(virtual_end_time[j-1], temp_machine_release_time[j][id_]) + \
                                                  virtual_job_time[j]

                    else:
                        virtual_release_time[j] = max(
                            max(virtual_end_time[j-1], temp_machine_release_time[j][id_]) + virtual_job_time[j],
                            min(temp_machine_release_time[j+1]))
                        virtual_end_time[j] = virtual_release_time[j]
                ##计算gamma
                    gamma = gamma + self.numstage / ((j + 1) + len(order_list[fac]) * (self.numstage - j - 1) / (self.numjob - 2)) * (
                                virtual_release_time[j] - temp_machine_release_time[j][id_] - virtual_job_time[j])
                fjk.append((self.numjob - len(order_list[fac]) - 2) * dalta + gamma)##按照原文的理解
                #fjk.append((self.numjob - k - 2) * dalta + gamma)
            ind = fjk.index(min(fjk))
            order_list[fac].append(order[ind])
            ##重新计算machine_release_time
            for j in range(self.numstage):
                id_ = np.argmin(all_machine_complete_time[fac][j])
                if j == 0:
                    end_time[order[ind] - 1, j] = all_machine_complete_time[fac][j][id_] + self.process_time[order[ind] - 1][j]
                    all_machine_complete_time[fac][j][id_] = max(end_time[order[ind] - 1, j],
                                                            min(all_machine_complete_time[fac][j + 1]))
                    end_time[order[ind] - 1, j] = all_machine_complete_time[fac][j][id_]

                elif j == self.numstage - 1:
                    all_machine_complete_time[fac][j][id_] = end_time[order[ind] - 1, j - 1] + self.process_time[order[ind] - 1, j]
                else:
                    all_machine_complete_time[fac][j][id_] = max(
                        max(end_time[order[ind] - 1, j - 1], all_machine_complete_time[fac][j][id_]) + self.process_time[
                            order[ind] - 1, j], min(all_machine_complete_time[fac][j + 1]))
                    end_time[order[ind] - 1, j] = all_machine_complete_time[fac][j][id_]
            
            ##计算makespan
            makespan_list[fac] = max(all_machine_complete_time[fac][self.numstage-1])
            del order[ind]
        ##增加最后一个工件，并计算MAKESPAN
        fac = np.argmin(makespan_list)
        order_list[fac].append(order[-1])
        for j in range(self.numstage):
            id_ = np.argmin(all_machine_complete_time[fac][j])
            if j == 0:
                end_time[order[-1] - 1, j] = all_machine_complete_time[fac][j][id_] + \
                                              self.process_time[order[-1] - 1][j]
                all_machine_complete_time[fac][j][id_] = max(end_time[order[-1] - 1, j],
                                                             min(all_machine_complete_time[fac][j + 1]))
                end_time[order[-1] - 1, j] = all_machine_complete_time[fac][j][id_]

            elif j == self.numstage - 1:
                all_machine_complete_time[fac][j][id_] = end_time[order[-1] - 1, j - 1] + self.process_time[
                    order[-1] - 1, j]
            else:
                all_machine_complete_time[fac][j][id_] = max(
                    max(end_time[order[-1] - 1, j - 1], all_machine_complete_time[fac][j][id_]) + self.process_time[
                        order[-1] - 1, j], min(all_machine_complete_time[fac][j + 1]))
                end_time[order[-1] - 1, j] = all_machine_complete_time[fac][j][id_]

        ##计算makespan
        makespan_list[fac] = max(all_machine_complete_time[fac][self.numstage-1])
        return order_list, makespan_list

    def NEH_LPT(self,order):
        sum_t = np.sum(self.process_time, axis=1)
        order = sorted(order, reverse=True, key=lambda k: sum_t[k - 1])
        makespan = self.fitness(order)
        # print('orderLPT: ' + str(order))
        # print('makespan ' + str(makespan))
        new_order, id_, makespan = self.NEH_step2(order)
        return new_order, id_, makespan

    def NEH_SPT(self,order):
        sum_t = np.sum(self.process_time, axis=1)
        order = sorted(order, key=lambda k: sum_t[k - 1])
        makespan = self.fitness(order)
        # print('orderSPT: ' + str(order))
        # print('makespan ' + str(makespan))
        new_order, id_, makespan = self.NEH_step2(order)
        return new_order, id_, makespan

    def NEH_SPT_first(self,order):
        sum_t_f = np.sum(self.process_time[:, :self.numstage - 1], axis=1)
        order = sorted(order, key=lambda k: sum_t_f[k - 1])
        makespan = self.fitness(order)
        # print('orderfirst: ' + str(order))
        # print('makespan ' + str(makespan))
        new_order, id_, makespan = self.NEH_step2(order)
        return new_order, id_, makespan

    def NEH_SPT_last(self,order):
        sum_t_l = np.sum(self.process_time[:, 1:], axis=1)
        order = sorted(order, key=lambda k: sum_t_l[k - 1])
        makespan = self.fitness(order)
        # print('orderlast: ' + str(order))
        # print('makespan ' + str(makespan))
        new_order, id_, makespan = self.NEH_step2(order)
        return new_order, id_, makespan

    def NEH2_step2(self,order):
        order_list = [[] for i in range(self.numfactory)]
        order_list[0].append(order[0])
        left_order = order[1:]
        makespan = [0 for i in range(self.numfactory)]
        makespan[0] = np.sum(self.process_time[order[0] - 1, :])
        for j in range(len(left_order)):
            order_list, makespan = self.job_insertion2(order_list, left_order[j], makespan)

        return order_list, makespan

    def NEH_step2(self,order):
        ##NEH step 2
        new_order = [order[0]]
        left_order = order[1:]
        for j in range(len(left_order)):
            new_order, id_, makespan = self.job_insertion(new_order, left_order[j])
        return new_order, id_, makespan

    def job_insertion2(self,order_list, insert_job, makespan):
        order_store = []
        makespan_max = []  # 包含每个组合list所有车间中max的makespan
        makespan_all = []  # 包含每个车间的makespan
        for f in range(self.numfactory):
            makespan_temp = copy.copy(makespan)
            order_f = copy.copy(order_list)
            for i in range(len(order_list[f]) + 1):
                if i == 0:
                    order_temp = [insert_job] + order_list[f]

                elif i == len(order_list[f]):
                    order_temp = order_list[f] + [insert_job]
                else:
                    order_temp = order_list[f][:i] + [insert_job] + order_list[f][i:]
                ##计算makespan
                makespan_temp[f] = self.fitness(order_temp)
                ms = max(makespan_temp)
                makespan_max.append(ms)
                makespan_all.append(copy.copy(makespan_temp))
                order_f[f] = order_temp
                ##存储order_temp
                order_store.append(copy.deepcopy(order_f))

        ##选择makespan最大的new_order
        id_ = np.argmin(makespan_max)
        order_list = order_store[id_]
        makespan = makespan_all[id_]
        return order_list, makespan

    def job_insertion(self,order, insert_job):
        makespan_temp = np.zeros(len(order) + 1)
        order_store = []
        for i in range(len(order) + 1):
            if i == 0:
                order_temp = [insert_job] + order

            elif i == len(order):
                order_temp = order + [insert_job]
            else:
                order_temp = order[:i] + [insert_job] + order[i:]
            ##计算makespan
            makespan_temp[i] = self.fitness(order_temp)
            ##存储order_temp
            order_store.append(order_temp)

        ##选择makespan最大的new_order
        id_ = np.argmin(makespan_temp)
        order = order_store[id_]
        makespan = min(makespan_temp)
        return order, id_, makespan

    #根据NEH2E_ex，需要知道第一次的插入后的位置和车间
    def job_insertion3(self, order_list, insert_job, makespan):
        fac_order_store = []
        fac_makespan_max = []  # 包含每个组合list所有车间中max的makespan
        fac_makespan_all = []  # 包含每个车间最好list的makespan
        id_position = []
        for f in range(self.numfactory):
            order_store = []
            makespan_max = []  # 包含单个车间每个组合list中max的makespan
            makespan_all = []
            makespan_temp = copy.copy(makespan)
            order_f = copy.copy(order_list)
            for i in range(len(order_list[f]) + 1):
                if i == 0:
                    order_temp = [insert_job] + order_list[f]

                elif i == len(order_list[f]):
                    order_temp = order_list[f] + [insert_job]
                else:
                    order_temp = order_list[f][:i] + [insert_job] + order_list[f][i:]
                ##计算makespan
                makespan_temp[f] = self.fitness(order_temp)
                ms = max(makespan_temp)
                makespan_max.append(ms)
                makespan_all.append(copy.copy(makespan_temp))
                order_f[f] = order_temp
                ##存储order_temp
                order_store.append(copy.deepcopy(order_f))
            ##选择车间内makespan最大的new_order
            id_ = np.argmin(makespan_max)
            id_position.append(id_)
            fac_order_store.append(order_store[id_])
            fac_makespan_all.append(makespan_all[id_])
            fac_makespan_max.append(makespan_max[id_])

        ##选择makespan最大的new_order
        best_fac_ = np.argmin(fac_makespan_max)
        new_order_list = fac_order_store[best_fac_]
        makespan = fac_makespan_all[best_fac_]
        position = id_position[best_fac_]
        return new_order_list, makespan,best_fac_,position

    def fitness(self, order):
        #process_time = copy.copy(self.process_time)
        ##工序开始时间，结束时间，blocking时间
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
                if j == 0:
                    start_end_block_time[item - 1, j * 3] = machine_complete_time[j][id_]
                    start_end_block_time[item - 1, j * 3 + 1] = start_end_block_time[item - 1, j * 3] + \
                                                                self.process_time[item - 1][j]
                    machine_complete_time[j][id_] = max(start_end_block_time[item - 1, j * 3 + 1],
                                                            min(machine_complete_time[j + 1]))
                    start_end_block_time[item - 1, j * 3 + 2] = machine_complete_time[j][id_] - start_end_block_time[item - 1, j * 3 + 1]
                    start_end_block_time[item - 1, j * 3 + 1] = machine_complete_time[j][id_]

                elif j == self.numstage - 1:
                    start_end_block_time[item - 1, j * 3] = start_end_block_time[item - 1, (j - 1) * 3 + 1]
                    start_end_block_time[item - 1, j * 3 + 1] = \
                        start_end_block_time[item - 1, j * 3] + self.process_time[item - 1, j]
                    machine_complete_time[j][id_] = start_end_block_time[item - 1, j * 3 + 1]
                else:

                    start_end_block_time[item - 1, j * 3] = start_end_block_time[item - 1, (j - 1) * 3 + 1]
                    start_end_block_time[item - 1, j * 3 + 1] = \
                        start_end_block_time[item - 1, j * 3] + self.process_time[item - 1, j]
                    machine_complete_time[j][id_] = max(start_end_block_time[item - 1, j * 3 + 1],
                                                            min(machine_complete_time[j + 1]))
                    start_end_block_time[item - 1, j * 3 + 2] = machine_complete_time[j][id_] - start_end_block_time[
                        item - 1, j * 3 + 1]
                    start_end_block_time[item - 1, j * 3 + 1] = machine_complete_time[j][id_]

        fitness = max(start_end_block_time[:,-2])
        return fitness

    def makespan_chrome(self,order_list):
        makespan_list = []
        for temp in order_list:
            if temp:
                makespan_list.append(self.fitness(temp))
            else:
                makespan_list.append(0)

        return makespan_list
