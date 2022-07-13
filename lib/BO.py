import GPyOpt
import numpy as np 
import os
from copy import deepcopy
from timeit import default_timer as timer
import scipy.cluster.hierarchy as sch
from scipy import stats
from scipy import spatial

class BayesianAgent:
	def __init__(self, env, features=None, args=None, tfwriter=None):
		self.env=env
		self.features=features
		self.constraints=None
		self.epoch=0
		self.max_iter=args.max_iter
		self.initial_points=args.initial_points
		self.args=args
		self.tfwriter=tfwriter
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
			Z = sch.linkage(disMat,method='ward')
			clusters = sch.fcluster(Z,args.n_clusters,'maxclust')-1
			self.groups=clusters.tolist()
		else:#无聚类
			self.groups = list(range(len(self.features)))
		print('groups: {}'.format(self.groups))
		self.domain = []
		for n in np.unique(self.groups):
			self.domain.append({'name': 'var_'+str(n), 'type': 'continuous', 'domain':(self.args.lbound, self.args.rbound)},)
	
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
		print('ratio list: {}'.format([float('{:.3f}'.format(ir)) for ir in ratio_list]))
		return ratio_list

	def objectfunc(self, x):
		self.epoch += 1
		print('Epoch: {}'.format(self.epoch))
		sample_val = x[0]#格式
		ratio_list = self.__generate_ratio_list(sample_val)#生成的ratio list
		reward, reserve_ratio = self.prune_bayesian(ratio_list)
		return self.__score(reward, reserve_ratio)

	def prune_bayesian(self, ratio_list):
		#state = np.array(self.env.layer_embedding)[None, :]
		self.env.reset()
		best_eval = False
		
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
		# Recording
		if(best_flag):
			self.best_epoch = self.epoch
			self.best_info = info
			print('Best epoch: {}, info {}'.format(self.best_epoch, self.best_info), file=open(os.path.join(self.args.output, 'bests.txt'), 'a'))
		print('Result : acc1={:.2f}, acc5={:.2f}, reserve ratio={:.2f}, flops={:.2f} M \n'.format(info['acc1'], info['acc5'], info['compress_ratio'], info['flops']))
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
		print([float('{:.3f}'.format(ir)) for ir in info['strategy']], file=open(os.path.join(self.args.output, 'result_strategy.txt'), 'a'))
		print(time_epoch, file=open(os.path.join(self.args.output, 'result_time.txt'), 'a'))	

		return reward, info['compress_ratio']

	def bayesianOptimize(self):
		self.start = timer()
		feasible_region = GPyOpt.Design_space(space = self.domain, constraints = self.constraints)
		#np.random.seed(123456)
		initial_design = GPyOpt.experiment_design.initial_design('sobol', feasible_region, self.initial_points)# random & latin & sobol
		print('initial points: {}'.format(self.initial_points))
		print('initial_design: {}'.format(initial_design))
		objective = GPyOpt.core.task.SingleObjective(self.objectfunc)
		model = GPyOpt.models.GPModel(exact_feval=False, optimize_restarts=5, verbose=False)
		aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
		acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)
		evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
		myBopt = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design, de_duplication = True)
		myBopt.run_optimization(max_iter=self.max_iter)

		for data_y in myBopt.Y:
			print(data_y[0], file=open(os.path.join(self.args.output, 'result_Y.txt'), 'a'))		
		#best_epoch = np.argmin(myBopt.Y)
		
		return self.best_epoch, self.best_info#产生最佳的ratio list