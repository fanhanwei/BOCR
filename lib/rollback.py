import GPyOpt
import numpy as np 
import os
from timeit import default_timer as timer
from lib.fifo import FIFO
import scipy.cluster.hierarchy as sch
from scipy import stats
from scipy import spatial

class BayesianAgent:
    def __init__(self, env, features=None, args=None, tfwriter=None):
        self.env=env
        self.args=args
        self.tfwriter=tfwriter
        self.initial_points=args.initial_points
        self.features=features
        self.constraints = None
        self.acq = 'EI'
        self.jitter = 0.01
        self.epoch=0
        self.best_epoch=0
        self.best_info=None
        self.acc_cache=[0,0,0,0,0,0,0,0,0,0]
        if self.args.model == 'mobilenetv2':
            self.fifo=FIFO(10,len(env.layer_embedding)-18)
        else:
            self.fifo=FIFO(10,len(env.layer_embedding)-2)
        self.best_num = 0
        self.new_X=[]
        self.tolerance = 0
        self.monitor = False
        self.monitor2 = False

        if self.args.static_cluster == True:#激活静态聚类
            if self.args.simlarity == 'ST':
                disMat = sch.distance.pdist(self.features,'euclidean')
            else:
                distributions = []
                X_range = np.mgrid[-15:15:1000j]
                positions = np.vstack([X_range.ravel()])
                for imp in self.env.importance_list:
                    shifted = imp - imp.median()
                    kernel = stats.gaussian_kde(shifted.cpu().numpy())
                    kernel.set_bandwidth(bw_method=kernel.factor / 1.)
                    distrib = np.reshape(kernel(positions).T, X_range.shape)
                    distributions.append(distrib+9.88131292e-50)#add a small value to avoid case 1/0
                if self.args.simlarity == 'EU':
                    eu_list = []
                    for i in range(len(distributions)-1):
                        for j in range(len(distributions)-1-i):
                            norm2 = np.linalg.norm(distributions[i]-distributions[j+i+1])
                            eu_list.append(norm2)
                    disMat = np.array(eu_list)
                elif self.args.simlarity == 'JS':
                    js_list = []
                    for i in range(len(distributions)-1):
                        for j in range(len(distributions)-1-i):
                            js = spatial.distance.jensenshannon(distributions[i], distributions[j+i+1])
                            js_list.append(js)
                    disMat = np.array(js_list)
                else:
                    raise NotImplementedError
            self.Z = sch.linkage(disMat, method='ward')#进行层次聚类
            clusters = sch.fcluster(self.Z, self.args.n_clusters,'maxclust')-1
            self.groups=clusters.tolist()
        else:#无聚类
            self.groups = list(range(len(self.features)))
        print('groups: {}'.format(self.groups))
        self.domain = []
        for n in np.unique(self.groups):
            self.domain.append({'name': 'var_'+str(n), 'type': 'continuous', 'domain':(self.args.lbound, self.args.rbound)},)
        print('groups: {}, bounds: {}, {} \n'.format(self.groups, self.args.lbound, self.args.rbound), file=open(os.path.join(self.args.output, 'final.txt'), 'a'))
        
    def __score(self, accuracy, size):
        error_penalty = -accuracy
        return error_penalty

    def __generate_ratio_list(self, sample_val):
        ratio_list = []
        ratio_list.append(1)
        print(self.groups, file=open(os.path.join(self.args.output, 'groups.txt'), 'a'))
        for n in self.groups:
            ratio_list.append(sample_val[n])
        ratio_list.append(1)
        print('ratio list: {}'.format(ratio_list))
        return ratio_list

    def objectfunc(self, x):
        self.epoch += 1
        print('Epoch {} :'.format(self.epoch))
        sample_val = x[0]#格式
        #self.n_clusters = len(sample_val)
        ratio_list = self.__generate_ratio_list(sample_val)#生成的ratio list
        reward, reserve_ratio = self.prune_bayesian(ratio_list)
        return self.__score(reward, reserve_ratio)

    def prune_bayesian(self, ratio_list):
        #state = np.array(self.env.layer_embedding)[None, :]
        self.env.reset()
        best_flag = False
        if self.args.model == 'mobilenetv2':
            for redundant_idx in [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]:
                ratio_list.insert(redundant_idx, 1)
            #print('fullfilled ratiolist: ', ratio_list)
        while True:
            layer_idx = self.env.cur_ind
            #print('    layer: {}'.format(layer_idx))
            action = ratio_list[layer_idx]
            #print('    Bayesian choosed preserve ratio: {}'.format(action))
            reward, done, info, best_flag = self.env.step_BO(action)
            #print('    Actual preserve ratio of the {} layer: {}'.format(layer_idx, self.env.strategy[-1]))
            if done:
                time_epoch = timer() - self.start
                break
        
        if self.args.model == 'mobilenetv2':
            redundant_list = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
            info['strategy'] = np.delete(info['strategy'], redundant_list, 0).tolist()
            print(info['strategy'], file=open(os.path.join(self.args.output, 'temp.txt'), 'a'))
        
        if info['compress_ratio'] > self.args.preserve_ratio-0.01 and info['compress_ratio'] < self.args.preserve_ratio+0.01:
            min_idx = np.argmin(self.acc_cache)
            if self.acc_cache[min_idx] < info[self.args.acc_metric]:
                self.acc_cache.pop(min_idx)
                self.acc_cache.append(info[self.args.acc_metric])
                self.fifo._update(np.array(info['strategy'][1:-1]).reshape(len(info['strategy'][1:-1]),1), min_idx)
                print('\n',self.acc_cache,'\n',self.fifo._items(), file=open(os.path.join(self.args.output, 'acc_cache.txt'), 'a'))
                print('cache updated')
        self.new_X.append(info['strategy'][1:-1])

        if self.epoch == 80:
            self.monitor = True
        if self.epoch == 130:
            self.monitor2 = True
        
        if  self.monitor and (self.epoch == 100 or self.tolerance >= 20):
            print('\n Go back!')
            self.monitor = False
            self.tolerance = 0
            #roll back
            clusters = sch.fcluster(self.Z, self.args.bridge_stage,'maxclust')-1
            self.groups=clusters.tolist()
            print('groups: {}'.format(self.groups))
            self.raise_dim()

            self.args.lbound = self.fifo._items().min()
            self.args.rbound = self.fifo._items().max()
            print('\n new bounds: ', self.args.lbound, self.args.rbound, '\n')
            print('epoch:', self.epoch, 'groups', self.groups, 'bounds: ', self.args.lbound, self.args.rbound, '\n', file=open(os.path.join(self.args.output, 'final.txt'), 'a'))
            
            self.domain = []
            for n in set(self.groups):
                self.domain.append({'name': 'var_'+str(n), 'type': 'continuous', 'domain':(self.args.lbound, self.args.rbound)},)
            feasible_region = GPyOpt.Design_space(space = self.domain, constraints = self.constraints)
            self.bo.space = feasible_region
            self.bo.model = GPyOpt.models.GPModel(exact_feval=False, optimize_restarts=5, verbose=False)
            #self.bo.model.model = None
            aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
            if self.acq == 'EI':
                self.bo.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.bo.model, feasible_region, optimizer=aquisition_optimizer, jitter=self.jitter)
            elif self.acq == 'PI':
                self.bo.acquisition = GPyOpt.acquisitions.AcquisitionMPI(self.bo.model, feasible_region, optimizer=aquisition_optimizer, jitter=self.jitter)
            else:
                raise RuntimeError('Undefined acquisition function: {}'.format(self.acq))
            self.bo.evaluator = GPyOpt.core.evaluators.Sequential(self.bo.acquisition)

        if  self.monitor2 and (self.epoch == 150 or self.tolerance >= 20):
            print('\n Go back all!')
            self.monitor2 = False
            #roll back
            self.groups = list(range(len(self.groups)))
            self.args.lbound = self.fifo._items().min()
            self.args.rbound = self.fifo._items().max()
            print('\n new bounds: ', self.args.lbound, self.args.rbound, '\n')
            print('epoch:', self.epoch, 'groups', self.groups, 'bounds: ', self.args.lbound, self.args.rbound, '\n', file=open(os.path.join(self.args.output, 'final.txt'), 'a'))
            
            self.domain = []
            for n in self.groups:
                self.domain.append({'name': 'var_'+str(n), 'type': 'continuous', 'domain':(self.args.lbound, self.args.rbound)},)
            feasible_region = GPyOpt.Design_space(space = self.domain, constraints = self.constraints)
            self.bo.space = feasible_region
            self.bo.model = GPyOpt.models.GPModel(exact_feval=False, optimize_restarts=5, verbose=False)
            #self.bo.model.model = None
            aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
            if self.acq == 'EI':
                self.bo.acquisition = GPyOpt.acquisitions.AcquisitionEI(self.bo.model, feasible_region, optimizer=aquisition_optimizer, jitter=self.jitter)
            elif self.acq == 'PI':
                self.bo.acquisition = GPyOpt.acquisitions.AcquisitionMPI(self.bo.model, feasible_region, optimizer=aquisition_optimizer, jitter=self.jitter)
            else:
                raise RuntimeError('Undefined acquisition function: {}'.format(self.acq))
            self.bo.evaluator = GPyOpt.core.evaluators.Sequential(self.bo.acquisition)
            self.bo.X = np.array(self.new_X)
        
        if best_flag:
            self.tolerance = 0
            self.best_epoch = self.epoch
            self.best_info = info
            print('Best epoch: {}, info {}'.format(self.best_epoch, self.best_info), file=open(os.path.join(self.args.output, 'bests.txt'), 'a'))
        else:
            self.tolerance += 1
            print('acc1={}, acc5={}, reserve ratio={} \n'.format(info['acc1'], info['acc5'], info['compress_ratio']))
        self.tfwriter.add_scalar('reward/last', reward, self.epoch)
        self.tfwriter.add_scalar('top5/best', self.env.best_top5, self.epoch)
        self.tfwriter.add_scalar('top1/best', self.env.best_top1, self.epoch)
        self.tfwriter.add_scalar('info/acc1', info['acc1'], self.epoch)
        self.tfwriter.add_scalar('info/acc5', info['acc5'], self.epoch)
        self.tfwriter.add_scalar('info/compress_ratio', info['compress_ratio'], self.epoch)
        self.tfwriter.add_text('info/best_policy', str(self.env.best_strategy), self.epoch)
        for i, preserve_rate in enumerate(info['strategy']):# record the preserve rate for each layer
            self.tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, self.epoch)
        print(info['acc1'], file=open(os.path.join(self.args.output, 'result_top1.txt'), 'a'))
        print(info['acc5'], file=open(os.path.join(os.path.join(self.args.output, 'result_top5.txt')), 'a'))
        print(info['compress_ratio'], file=open(os.path.join(self.args.output, 'result_reserve_ratio.txt'), 'a'))
        print(info['strategy'], file=open(os.path.join(self.args.output, 'result_strategy.txt'), 'a'))
        print(time_epoch, file=open(os.path.join(self.args.output, 'result_time.txt'), 'a'))	

        return reward, info['compress_ratio']

    def bayesianOptimize(self):
        self.start = timer()
        feasible_region = GPyOpt.Design_space(space = self.domain, constraints = self.constraints)
        initial_design = GPyOpt.experiment_design.initial_design('sobol', feasible_region, self.initial_points)# random & latin & sobol
        print('initial points: {}'.format(self.initial_points))
        print('initial_design: {}'.format(initial_design))
        objective = GPyOpt.core.task.SingleObjective(self.objectfunc)
        model = GPyOpt.models.GPModel(exact_feval=False, optimize_restarts=5, verbose=False)
        aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
        if self.acq == 'EI':
            acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer, jitter=self.jitter)
        elif self.acq == 'PI':
            acquisition = GPyOpt.acquisitions.AcquisitionMPI(model, feasible_region, optimizer=aquisition_optimizer, jitter=self.jitter)
        else:
            raise RuntimeError('Undefined acquisition function: {}'.format(self.acq))
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        myBopt = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design, de_duplication = True)
        self.bo=myBopt
        myBopt.run_optimization(max_iter=self.args.max_iter)
        for data_x in myBopt.X:
            print(data_x, file=open(os.path.join(self.args.output,'result_X.txt'), 'a'))
        for data_y in myBopt.Y:
            print(data_y, file=open(os.path.join(self.args.output, 'result_Y.txt'), 'a'))
        #best_epoch = np.argmin(myBopt.Y)
        
        return self.best_epoch, self.best_info#产生最佳的ratio list

    def raise_dim(self):
        print(self.bo.X, file=open(os.path.join(self.args.output, 'X'+str(self.epoch)+'.txt'), 'a'))
        max=0
        _dict={}
        del_buffer = []
        new_group=[]
        for i, n in enumerate(self.groups):
            if n in _dict.keys():
                new_group.append(_dict[n])
                del_buffer.append(i)
            else:
                _dict[n] = max
                new_group.append(max)
                max += 1
        print("new_group: ",new_group)
        self.bo.X = np.delete(self.new_X, del_buffer, axis=1)
        self.groups = new_group
