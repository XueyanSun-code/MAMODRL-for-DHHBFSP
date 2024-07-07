import time

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from Initialization import Initializations

plt.style.use('seaborn-v0_8')

class BigroupGeneticAlgorithm():
    def __init__(self, ND,ps,sr1,sr2,mr,rate_high,numfactory, numjob, numstage, nummachine, process_time):
        self.ND = ND
        self.population_size = ps
        self.sr1 = sr1
        self.sr2 = sr2
        self.mr = mr
       # self.iteration = iteration
        self.rate_high = rate_high
        self.numfactory = numfactory
        self.numjob = numjob
        self.numstage = numstage
        self.nummachine = nummachine
        self.process_time = process_time

    def BGA_running(self):
        ###可以减少elite group个体，降低随机性，至少小于一般种群
        t0 = time.time()
        initial = Initializations(self.numfactory, self.numjob, self.numstage, self.nummachine, self.process_time)

        order = list(range(1, self.numjob + 1))
        population, makespan_pop = initial.population_initial(order, self.population_size)
        makespan_max = [max(l) for l in makespan_pop]
        best_index = np.argmin(makespan_max)
        termination_criterion = 20 * self.numjob * self.numstage * self.numfactory / 1000  ##毫秒
        ## Perform PDLS on chrome𝑏𝑒𝑠𝑡 Problem-dependent local search
        population[best_index], makespan_pop[best_index] = self.VND_LS_9_2(population[best_index], makespan_pop[best_index],t0,termination_criterion)
        best_chrome = copy.deepcopy(population[best_index])
        best_makespan = copy.copy(makespan_pop[best_index])
        makespan_max = [max(l) for l in makespan_pop]
        # print(str(best_chrome) + str(best_makespan))
        ##这里开始迭代循环
        runtime = time.time() - t0

        # print(str(termination_criterion))
        while runtime < termination_criterion:

            ##GAgroup split

            high_group = []
            high_makespan = []
            ##选择好的解留下，按照makespan分成两个高低两个group
            high_group_size = int(self.population_size *self.rate_high)
            for i in range(high_group_size):
                index = makespan_max.index(min(makespan_max))
                high_group.append(population[index])
                high_makespan.append(makespan_pop[index])
                del population[index]
                del makespan_pop[index]
                del makespan_max[index]
            ##剩余个体成为lowgroup
            low_group = copy.deepcopy(population)
            low_makespan = copy.deepcopy(makespan_pop)
            #selection占据highgroup的25%，lowgroup的5%，向上取整,选择出的个体直接保留到下一代,精英选择
            population = []
            makespan_pop = []
            ##这里的selection是在high group随机选择
            high_selection = int(np.ceil(self.sr1 * high_group_size))
            for i in range(high_selection):
                rand = np.random.randint(high_group_size)
                population.append(high_group[rand])
                makespan_pop.append(high_makespan[rand])
            low_selection = int(np.ceil(self.sr2 * len(low_group)))
            for i in range(low_selection):
                rand = np.random.randint(len(low_group))
                population.append(low_group[rand])
                makespan_pop.append(low_makespan[rand])
            ##交叉算子
            ##分三部分，两个组内交叉，组之间交叉
            ##随机选择两个个体

            new_child_group_1 = self.same_crossover(high_group)
            new_child_group_2 = self.same_crossover(low_group)
            new_child_group_3 = self.different_crossover(high_group, low_group)
            ##变异算子（父代个体的变异）分为插入和交换
            ##rand<0.5 插入，否则交换
            new_child_group_4 = self.mutation_group(high_group, low_group)
            # generate 4 groups children, unit and selection
            child_group = new_child_group_1 + new_child_group_2 + new_child_group_3 + new_child_group_4
            # 父代子带只保留一个（暂不考虑，可以在交叉过程中直接代替），其他则是锦标赛选择
            ##每次随机选三个，然后makespam最小的留下
            left_size = self.population_size - len(population)
            left_pop, left_make = self.children_selection(child_group, left_size)
            ##合并
            population = population + left_pop
            makespan_pop = makespan_pop + left_make
            makespan_max = [max(l) for l in makespan_pop]
           ##这里采用了DDE里的Elitist retain strategy，让最优解永远保留
            best_id_temp = np.argmin(makespan_max)
            population[best_id_temp],makespan_pop[best_id_temp] = self.DR(population[best_id_temp],makespan_pop[best_id_temp], self.ND)
            # ## Perform PDLS on chrome𝑏𝑒𝑠𝑡 Problem-dependent local search
            population[best_id_temp],makespan_pop[best_id_temp] = self.VND_LS_9_2(population[best_id_temp],makespan_pop[best_id_temp],t0,termination_criterion)
            makespan_max = [max(l) for l in makespan_pop]
            best_id_ = np.argmin(makespan_max)
            if max(makespan_pop[best_id_]) < max(best_makespan):
                best_chrome = copy.deepcopy(population[best_id_])
                best_makespan = copy.copy(makespan_pop[best_id_])
            else:
                ##随即代替一个个体
                rand = np.random.randint(self.population_size)
                population[rand] = best_chrome
                makespan_pop[rand] = best_makespan
                makespan_max = [max(l) for l in makespan_pop]
            runtime = time.time() - t0
            #print(best_chrome, best_makespan)
            #print(str(best_chrome) + str(best_makespan))
        return best_chrome, best_makespan

    def children_selection(self,child_group, left_size, ):##这里可以改为rand<0.5,makespan最小；>0.5 potential 最好
        tunament_num = 3
        left_pop = []
        left_make = []
        for i in range(left_size):
            small_group = []
            make_list_group = []
            makespan_group = []
            for j in range(tunament_num):
                rand = np.random.randint(len(child_group))
                make_list = self.makespan_chrome(child_group[rand])
                x = max(make_list)
                small_group.append(child_group[rand])
                make_list_group.append(make_list)
                makespan_group.append(x)
            if random.random()<0.5:
                s = np.argmin(makespan_group)
                left_pop.append(small_group[s])
                left_make.append(make_list_group[s])
            else:##potential
                ##进行三个局部搜索
                potential_rate = [0 for k in range(len(small_group))]
                for j,chrome in enumerate(small_group):
                    potential_rate[j] = self.simple_local_search(chrome,make_list_group[j])
                s = np.argmax(potential_rate)
                left_pop.append(small_group[s])
                left_make.append(make_list_group[s])
        return left_pop, left_make

    def Destruction(self,order_list, makespan_list,ds):
        index = [i for i, val in enumerate(makespan_list) if val == max(makespan_list)]
        extract_jobs = []
        ##随机选择一个关键工厂进行提取
        rand = np.random.randint(len(index))
        choosed_fac = index[rand]
        for i in range(int(ds/2)):
            if order_list[choosed_fac]:
                rand = np.random.randint(len(order_list[choosed_fac]))
                extract_jobs.append(order_list[choosed_fac][rand])
                del order_list[choosed_fac][rand]
            else:
                break
        if ds > len(extract_jobs):
            ##剩余工件在所有fac随机提取
            for i in range(ds - len(extract_jobs)):
                rand_fac = np.random.randint(self.numfactory)
                while len(order_list[rand_fac]) == 0:
                    rand_fac = np.random.randint(self.numfactory)
                rand_pos = np.random.randint(len(order_list[rand_fac]))
                extract_jobs.append(order_list[rand_fac][rand_pos])
                del order_list[rand_fac][rand_pos]
        return extract_jobs, order_list

    def reconstruction(self,extract_jobs, order_list):
        ##先计算makespan
        makespan = self.makespan_chrome(order_list)
        order_list, makespan_list = self.NEH2_en( order_list, extract_jobs, makespan)
        return order_list, makespan_list

    def NEH2_en(self, order_list,left_order,makespan_list):
        ##NEH step 2
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
                new_order,  makespan = self.job_insertion(new_order, left)
                order_list[best_fac_] = new_order
                makespan_list[best_fac_] = makespan

        return order_list, makespan_list

    def DR(self,pi,makespan_pi,d):
        pi1 = copy.deepcopy(pi)
        makespan_pi1 = copy.copy(makespan_pi)
        extract_jobs, pi1 = self.Destruction( pi1, makespan_pi1, d)

        pi1,makespan_pi1 = self.reconstruction(extract_jobs, pi1)
        return pi1,makespan_pi1

    def simple_local_search(self,chrome, make_list):
        # print('simple_local_search', chrome)
        index = []
        for k in range(3):
            if k == 0:
                list_, makespan_list = self.N1(chrome, make_list)
            elif k == 1:
                list_, makespan_list = self.N2(chrome, make_list)
            else:
                list_, makespan_list = self.N3(chrome, make_list)
            index.append(max(makespan_list))
        potential_rate = (max(index)-min(index))/max(make_list)
        return potential_rate

    def different_crossover(self,group1, group2):
        ##确定交叉次数
        new_child_group = []
        for i in range(int(len(group1) / 2)):
            ##随机选择两个个体
            rand1 = np.random.randint(len(group1))
            rand2 = np.random.randint(len(group2))
            p1 = group1[rand1]
            p2 = group2[rand2]
            c1, c2 = self.crossover_opration(p1, p2)
            new_child_group.append(c1)
            new_child_group.append(c2)
        return new_child_group

    def same_crossover(self,group):
        ##确定交叉次数
        new_child_group = []
        for i in range(int(len(group) / 2)):
            ##随机选择两个个体
            rand1 = np.random.randint(len(group))
            rand2 = np.random.randint(len(group))
            while rand1 == rand2:
                rand2 = np.random.randint(len(group))
            p1 = group[rand1]
            p2 = group[rand2]
            c1, c2 = self.crossover_opration(p1, p2)
            new_child_group.append(c1)
            new_child_group.append(c2)
        return new_child_group

    def crossover_opration(self,p1, p2):
        ##交叉操作有很多种，车间交叉，车间内交叉，所有车间内交叉，变成一个list进行交叉
        # 选择交叉的工厂
        c1 = copy.deepcopy(p1)
        c2 = copy.deepcopy(p2)
        r = np.random.randint(len(c1))
        temp1 = copy.copy(c1[r])
        temp2 = copy.copy(c2[r])
        c1[r] = copy.copy(temp2)
        c2[r] = copy.copy(temp1)
        ##找出他们相同的工件
        # print('crossover_operation_p1',p1)
        # print('crossover_operation_p2', p2)
        same_job = []
        for jobi in c1[r]:
            for jobj in c2[r]:
                if jobi == jobj:
                    same_job.append(jobi)
                    del temp1[temp1.index(jobi)]  ##删除后的工件是p1在交叉后缺少的工件，也是p2在交叉后重复的工件
                    del temp2[temp2.index(jobi)]  ##删除后的工件是p2在交叉后缺少的工件，也是p1在交叉后重复的工件
                    break
        ##首先应先删除其他不交叉车间的重复工件，再增加缺少的
        ##c1删除
        for i, item in enumerate(temp2):
            for f, flist in enumerate(c1):
                if f != r:
                    t_flist = copy.copy(flist)
                    for job in t_flist:
                        if item == job:
                            del flist[flist.index(job)]
                            break
                    else:
                        continue
                    break
        ##c2删除
        for i, item in enumerate(temp1):
            for f, flist in enumerate(c2):
                if f != r:
                    t_flist = copy.copy(flist)
                    for job in t_flist:
                        if item == job:
                            del flist[flist.index(job)]
                            break
                    else:
                        continue
                    break
        # c1增加
        makespan_list = self.makespan_chrome(c1)
        for i, item in enumerate(temp1):
            ##选择最小的makespan进行随机插入
            f = np.argmin(makespan_list)
            if c1[f]:
                rand = np.random.randint(len(c1[f]) + 1)
                if rand == 0:
                    c1[f] = [item] + c1[f]

                elif rand == len(c1[f]):
                    c1[f] = c1[f] + [item]
                else:
                    c1[f] = c1[f][:rand] + [item] + c1[f][rand:]

            else:
                c1[f] = [item]
            ##计算新工序的makespan
            makespan_list[f] = self.fitness(c1[f])
        # c2增加
        makespan_list = self.makespan_chrome(c2)
        for i, item in enumerate(temp2):
            ##选择最小的makespan进行随机插入
            f = np.argmin(makespan_list)
            if c2[f]:
                rand = np.random.randint(len(c2[f]) + 1)
                if rand == 0:
                    c2[f] = [item] + c2[f]

                elif rand == len(c2[f]):
                    c2[f] = c2[f] + [item]
                else:
                    c2[f] = c2[f][:rand] + [item] + c2[f][rand:]
                ##计算新工序的makespan
                makespan_list[f] = self.fitness(c2[f])
            else:
                c2[f] = [item]
            ##计算新工序的makespan
            makespan_list[f] = self.fitness(c2[f])
        return c1, c2

    def mutation_group(self,group1, group2):
        new_child_group = []
        group = group1 + group2
        for i, item in enumerate(group):
            if random.random() < self.mr:
                job_list = copy.deepcopy(item)
                new_child_group.append(self.mutation(job_list))
        return new_child_group

    def mutation(self,job_list):
        all_sequence = []
        for i in range(len(job_list)):
            all_sequence = all_sequence + job_list[i]
        rand1 = np.random.randint(len(all_sequence))
        if random.random() < 0.5:  ##插入
            temp = all_sequence[rand1]
            del all_sequence[rand1]
            rand2 = np.random.randint(len(all_sequence))
            if rand2 == 0:
                all_sequence = [temp] + all_sequence

            elif rand2 == len(all_sequence):
                all_sequence = all_sequence + [temp]
            else:
                all_sequence = all_sequence[:rand2] + [temp] + all_sequence[rand2:]
        else:
            temp = all_sequence[rand1]
            rand2 = np.random.randint(len(all_sequence))
            all_sequence[rand1] = all_sequence[rand2]
            all_sequence[rand2] = temp
        ##change back to job_list
        x = 0
        for i in range(len(job_list)):
            job_list[i] = all_sequence[x:x + len(job_list[i])]
            x = x + len(job_list[i])

        return job_list

    def VND9_2_N1(self, list, makespan_list,t0,standard ):  ##Max_min,工厂间进行insert（move）,每得到新的更好解后重新进行优化,max 与min之间
        pi1 = copy.deepcopy(list)
        makespan_pi1 = copy.copy(makespan_list)
        move = 1
        while move:
            t1 = time.time()
            if t1 - t0 >standard:
                break
            move = 0
            critical_f = np.argmax(makespan_pi1)
            ##循环关键factory
            for i, job in enumerate(pi1[critical_f]):

                temp_order = copy.copy(pi1[critical_f])
                del temp_order[i]
                makespan_order = self.fitness(temp_order)
                if makespan_order < makespan_pi1[critical_f]:
                    list_temp = []
                    makespan_temp = []
                    if len(temp_order) > 1:

                        temp_order, makespan_order = self.local_insertion_all(makespan_order,temp_order)
                    list_temp.append(temp_order)
                    makespan_temp.append(makespan_order)
                    ##选择最小factory
                    rand_f = np.argmin(makespan_pi1)
                    if rand_f == critical_f:
                        return pi1, makespan_pi1

                    temp_rand_order = copy.copy(pi1[rand_f])
                    temp_rand_order, makespan_rand, id = self.job_insertion_id(temp_rand_order, job)
                    ##如果插入之后的最好结果仍然大于现在关键工厂的值，则进行一下的操作：进行插入工件局部的前后调整
                    if makespan_rand > makespan_order:
                        if len(temp_rand_order) > 2:
                            temp_rand_order, makespan_rand = self.local_insertion_all(makespan_rand, temp_rand_order)
                    list_temp.append(temp_rand_order)
                    makespan_temp.append(makespan_rand)
                    if max(makespan_temp) < max(makespan_pi1):
                        pi1[critical_f] = copy.copy(list_temp[0])
                        pi1[rand_f] = copy.copy(list_temp[1])
                        makespan_pi1[critical_f] = makespan_temp[0]
                        makespan_pi1[rand_f] = makespan_temp[1]
                        move = 1
                        break
            if move == 0:
                break

        return pi1, makespan_pi1

    def VND9_2_N2(self, list, makespan_list,t0,standard):  ##工厂间进行swap,关键factory选择1个job加工时间最小
        pi1 = copy.deepcopy(list)
        makespan_pi1 = copy.copy(makespan_list)
        move = 1
        while move:
            t1 = time.time()
            if t1 - t0 > standard:
                break
            move = 0
            critical_f = np.argmax(makespan_pi1)
            ##随机选择一个factory

            rand_f = np.random.randint(self.numfactory)
            while rand_f == critical_f or len(pi1[rand_f]) == 0:
                if len(list) == 2:
                    return pi1, makespan_pi1
                rand_f = np.random.randint(self.numfactory)
            ##关键factory随机选择1个job


            j = np.random.randint(len(pi1[critical_f]))
            for i in range(len(pi1[rand_f])):
                list_temp = []
                makespan_temp = []
                # list_temp.append(temp_order)
                # makespan_temp.append(makespan_order)
                temp_order = copy.deepcopy(pi1[critical_f])
                temp_rand_order = copy.deepcopy(pi1[rand_f])
                temp_order[j] = temp_rand_order[i]
                temp_rand_order[i] = pi1[critical_f][j]
                makespan_order = self.fitness(temp_order)
                makespan_order_rand = self.fitness(temp_rand_order)
                ##两个order分别进行local insertion
                if len(temp_order) > 2:
                    temp_order, makespan_order = self.local_insertion_all(makespan_order, temp_order)
                if len(temp_rand_order) > 2:
                    temp_rand_order, makespan_order_rand = self.local_insertion_all(makespan_order_rand,
                                                                                temp_rand_order)
                list_temp.append(temp_order)
                list_temp.append(temp_rand_order)
                makespan_temp.append(makespan_order)
                makespan_temp.append(makespan_order_rand)
                if max(makespan_temp) < max(makespan_pi1):
                    pi1[critical_f] = copy.copy(list_temp[0])
                    pi1[rand_f] = copy.copy(list_temp[1])
                    makespan_pi1[critical_f] = makespan_temp[0]
                    makespan_pi1[rand_f] = makespan_temp[1]
                    move = 1
                    break
            if move == 0:
                break

        return pi1, makespan_pi1

    def VND_LS_9_2(self, order_list1, makespan_list1,t0,standard):
        order_list = copy.deepcopy(order_list1)
        makespan_list = copy.copy(makespan_list1)

        k = 0
        while k < 3:
            if k == 0:
                if time.time() -t0 < standard:
                    list_p, makespan_listp = self.VND9_2_N1(order_list, makespan_list,t0,standard)
                else:
                    break
            elif k == 1:
                if time.time() -t0 < standard:
                    list_p, makespan_listp = self.VND9_2_N2(order_list, makespan_list,t0,standard)
                else:
                    break
            else:
                if time.time() - t0 < standard:
                    list_p, makespan_listp = self.Ribas_LS(order_list, makespan_list)
                else:
                    break
            if max(makespan_listp) < max(makespan_list):
                order_list = list_p
                makespan_list = makespan_listp
                k = 0

            else:
                k += 1

        return order_list, makespan_list

    def Ribas_LS(self,new_order_list1,makespan_list1):
        order_list = copy.deepcopy(new_order_list1)
        makespan_list = copy.copy(makespan_list1)
        key_index = makespan_list.index(max(makespan_list))
        ##LS1,LS2，在工厂内进行局部搜索
        order = order_list[key_index]
        makespan = makespan_list[key_index]
        flag = True
        while flag:
            flag = False
            order_list[key_index],makespan_list[key_index],f1 = self.swap_inner(order_list[key_index],makespan_list[key_index])
            order_list[key_index],makespan_list[key_index],f2 = self.insertion_inner(order_list[key_index],makespan_list[key_index])
            if f1 or f2:
                flag = True
        return order_list,makespan_list

    def swap_inner(self,order1,makespan1):
        flag = False
        order2 = copy.copy(order1)
        makespan2 = copy.copy(makespan1)
        for i,job in enumerate(order1):
            if i == len(order1) - 1:
                break
            order_store = []
            makespan_store = []

            ind = order2.index(job)
            for j in range(ind+1,len(order2)):
                order = copy.copy(order2)
                order[ind] = order[j]
                order[j] = job
                makespan_temp = self.fitness(order)
                order_store.append(order)
                makespan_store.append(makespan_temp)

            id_ = np.argmin(makespan_store)
            order_ = order_store[id_]
            makespan_ = makespan_store[id_]
            if makespan_ < makespan2:
                order2 = copy.copy(order_)
                makespan2 = copy.copy(makespan_)
        if  makespan2 < makespan1:
            order1 = copy.copy(order2)
            makespan1 = copy.copy(makespan2)
            flag = True
        return order1, makespan1,flag

    def insertion_inner(self, order1,makespan1):
        flag = False
        order2 = copy.copy(order1)
        makespan2 = copy.copy(makespan1)
        for i, job in enumerate(order1):
            order = copy.copy(order2)
            del order[order.index(job)]
            order,makespan = self.job_insertion(order,job)
            if makespan < makespan2:
                order2 = copy.copy(order)
                makespan2 = copy.copy(makespan)
        if makespan2 < makespan1:
            order1 = copy.copy(order2)
            makespan1 = copy.copy(makespan2)
            flag = True
        return order1, makespan1,flag

    def local_insertion(self, makespan_rand, temp_rand_order, id):
        if id < 2 and (len(temp_rand_order) - id - 1) > 1:  # after
            # 确保前面的工件少于2且后面的大于等于2，调整后面的工件顺序,删除后面的工件，id 不变
            improved = True
            while improved:
                temp_rand_order1 = copy.copy(temp_rand_order)
                r_position = random.randint(id + 1, len(temp_rand_order1) - 1)
                job1 = temp_rand_order1[r_position]
                del temp_rand_order1[r_position]
                temp_rand_order1, makespan_rand_new = self.job_insertion_part(temp_rand_order1, id, "after", job1)
                if makespan_rand_new < makespan_rand:
                    makespan_rand = makespan_rand_new
                    temp_rand_order = temp_rand_order1
                else:
                    improved = False
        elif id > 1 and (len(temp_rand_order) - id - 1) < 1:  # before
            # 确保前面的工件大于等于2且后面的小于2，调整前面的工件顺序，删除前面的工件，id -1
            improved = True
            while improved:
                temp_rand_order1 = copy.copy(temp_rand_order)
                r_position = random.randint(0, id - 1)
                #print(id, ' ', temp_rand_order1,' ', r_position)
                job1 = temp_rand_order1[r_position]
                del temp_rand_order1[r_position]
                temp_rand_order1, makespan_rand_new = self.job_insertion_part(temp_rand_order1, id, "before", job1)
                if makespan_rand_new < makespan_rand:
                    makespan_rand = makespan_rand_new
                    temp_rand_order = temp_rand_order1
                else:
                    improved = False
        elif id > 1 and (len(temp_rand_order) - id - 1) > 1:
            ##前后都有大于两个工件的，进行随机选择一侧进行
            if random.random() < 0.5:
                type = 'before'
            else:
                type = 'after'
            improved = True
            while improved:
                temp_rand_order1 = copy.copy(temp_rand_order)
                r_position = random.randint(0, id - 1)
                job1 = temp_rand_order1[r_position]
                del temp_rand_order1[r_position]
                temp_rand_order1, makespan_rand_new = self.job_insertion_part(temp_rand_order1, id,
                                                                              type, job1)
                if makespan_rand_new < makespan_rand:
                    makespan_rand = makespan_rand_new
                    temp_rand_order = temp_rand_order1
                else:
                    improved = False
        return temp_rand_order, makespan_rand

    def local_insertion_all(self, makespan_rand, temp_rand_order):

        # 确保前面的工件少于2且后面的大于等于2，调整后面的工件顺序,删除后面的工件，id 不变
        improved = True
        while improved:
            temp_rand_order1 = copy.copy(temp_rand_order)
            r_position = random.randint(0,len(temp_rand_order1)-1)
            job1 = temp_rand_order1[r_position]
            del temp_rand_order1[r_position]
            temp_rand_order1, makespan_rand_new = self.job_insertion(temp_rand_order1, job1)
            if makespan_rand_new < makespan_rand:
                makespan_rand = makespan_rand_new
                temp_rand_order = temp_rand_order1
            else:
                improved = False

        return temp_rand_order, makespan_rand


    def job_insertion_part(self, order, id, type, insert_job):
        if type == 'after':
            makespan_temp = []
            order_store = []
            for i in range(id + 1, len(order) + 1):

                if i == len(order):
                    order_temp = order + [insert_job]
                else:
                    order_temp = order[:i] + [insert_job] + order[i:]
                ##计算makespan
                makespan_temp.append(self.fitness(order_temp))
                ##存储order_temp
                order_store.append(order_temp)
        else:
            makespan_temp = []
            order_store = []
            for i in range(0, id):
                if i == 0:
                    order_temp = [insert_job] + order
                else:
                    order_temp = order[:i] + [insert_job] + order[i:]
                ##计算makespan
                makespan_temp.append(self.fitness(order_temp))
                ##存储order_temp
                order_store.append(order_temp)

        ##选择makespan最大的new_order
        id_ = np.argmin(makespan_temp)
        order = order_store[id_]
        makespan = min(makespan_temp)
        return order, makespan

    def job_insertion_id(self, order, insert_job):
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
        return order, makespan, id_
        # 关键工厂内进行

    def Problem_dependent_local_search(self,list1,makespan_list):
        iteration = 0
        while iteration < 10:
            k = 0
            while k <3:
                if k == 0:
                    list_p,makespan_listp = self.N1(list1,makespan_list)
                elif k == 1:
                    list_p,makespan_listp = self.N2(list1,makespan_list)
                else:
                    list_p,makespan_listp = self.N3(list1,makespan_list)
                if max(makespan_listp) < max(makespan_list):
                    list1 = list_p
                    makespan_list = makespan_listp
                k += 1
            iteration += 1
        return list1,makespan_list

    def N1(self,list,makespan_list): ##工厂间进行insert
        pi1 = copy.deepcopy(list)
        #print('N1-pi1', pi1)
        makespan_pi1 = copy.copy(makespan_list)
        critical_f = np.argmax(makespan_pi1)
        #print('critical_f',critical_f)
        ##随机选择一个factory
        rand_f = np.random.randint(self.numfactory)
        while rand_f == critical_f:
            rand_f = np.random.randint(self.numfactory)
        ##关键factory随机选择一个job
        #print(pi1)
        rand_pos_fc = np.random.randint(len(pi1[critical_f]))
        job = pi1[critical_f][rand_pos_fc]
        del pi1[critical_f][rand_pos_fc]
        pi1[rand_f],makespan_pi1[rand_f] = self.job_insertion(pi1[rand_f],job)
        makespan_pi1 = self.makespan_chrome(pi1)
        return pi1, makespan_pi1


    def N2(self,list,makespan_list):##工厂间进行swap
        pi1 = copy.deepcopy(list)
        makespan_pi1 = copy.copy(makespan_list)
        critical_f = np.argmax(makespan_pi1)
        ##随机选择一个factory
        rand_f = np.random.randint(self.numfactory)
        while rand_f == critical_f:
            rand_f = np.random.randint(self.numfactory)
        ##关键factory随机选择一个job
        rand_pos_fc = np.random.randint(len(pi1[critical_f]))
        for i in range(len(pi1[rand_f])):
            pi = copy.deepcopy(pi1)
            makespan_pi = copy.copy(makespan_pi1)
            pi[critical_f][rand_pos_fc] = pi[rand_f][i]
            pi[rand_f][i] = pi1[critical_f][rand_pos_fc]
            makespan_pi[critical_f] = self.fitness(pi[critical_f])
            makespan_pi[rand_f] = self.fitness(pi[rand_f])
            if max(makespan_pi) < max(makespan_pi1):
                pi1 = copy.deepcopy(pi)
                makespan_pi1 = copy.copy(makespan_pi)
                break

        return pi1, makespan_pi1

    def N3(self,list,makespan_list):##关键工厂内insert
        pi1 = copy.deepcopy(list)
        makespan_pi1 = copy.copy(makespan_list)
        critical_f = np.argmax(makespan_pi1)
        ##随机选择一个工件
        rand_pos_fc = np.random.randint(len(pi1[critical_f]))
        job = pi1[critical_f][rand_pos_fc]
        del pi1[critical_f][rand_pos_fc]
        pi1[critical_f], makespan_pi1[critical_f] = self.job_insertion(pi1[critical_f], job)
        return pi1, makespan_pi1

    def job_insertion(self, order, insert_job):
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
        return order, makespan

    def job_insertion2(self, order_list, insert_job, makespan):
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

        return new_order_list, makespan, best_fac_, position

    def fitness(self, order):
        # process_time = copy.copy(self.process_time)
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
                                                                self.process_time[item - 1, j]
                    machine_complete_time[j][id_] = max(start_end_block_time[item - 1, j * 3 + 1],
                                                        min(machine_complete_time[j + 1]))
                    start_end_block_time[item - 1, j * 3 + 2] = machine_complete_time[j][id_] - start_end_block_time[
                        item - 1, j * 3 + 1]
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

        fitness = max(start_end_block_time[:, -2])
        return fitness

    def makespan_chrome(self,order_list):
        makespan_list = []
        for temp in order_list:
            if temp:
                makespan_list.append(self.fitness(temp))
            else:
                makespan_list.append(0)

        return makespan_list

    def group_split(self,population_size,population,makespan_pop):
        ##GAgroup split
        makespan_max = [max(l) for l in makespan_pop]
        high_group = []
        high_makespan = []
        ##选择好的解留下，按照makespan分成两个高低两个group
        high_group_size = int(population_size / 2)
        for i in range(high_group_size):
            index = makespan_max.index(max(makespan_max))
            high_group.append(population[index])
            high_makespan.append(makespan_pop[index])
            del population[index]
            del makespan_pop[index]
        ##剩余个体成为lowgroup
        low_group = copy.deepcopy(population)
        low_makespan = copy.deepcopy(makespan_pop)



