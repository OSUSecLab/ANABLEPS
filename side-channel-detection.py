import os, commands,copy
import ptfileread
import json
import time
import argparse
import numpy


class PageNode(object):
	__slots__=('id', 'bb_addrs', 'contain_list', 'out_degree', 'in_degree', 'order_pattern', 'order', 'time_list', 'time_mean', 'time_std', 'successors', 'predcessors')
	def __init__(self, nid, bb_addrs = None, file_index = None, time_list = None, order_pattern = None, order = None):
		self.id = nid
		self.bb_addrs = set() if bb_addrs == None else set([bb_addrs])
		self.contain_list = set() if file_index == None else set([file_index])
		self.out_degree = 0
		self.in_degree = 0
		self.order_pattern = {}	if order_pattern is None else order_pattern	# key: next node, value: id
		self.order = {}	if order is None else order 	# key: file_id, value: order
		# self.time = {} if time is None else time	# key: file_id, vaule: time
		self.time_list = {file_index:[[]]}	if time_list is None else {file_index: [time_list]} # key: file_id, value: time order
		self.time_mean = {}	# key:file_id, value: mean list
		self.time_std = {}	# key:file_id, value: std list
		self.successors = []
		self.predcessors = []


	def __repr__(self):
		return '<PageNode id %#x>' %(self.id)


	def set_addr(self, addr):
		self.addr = addr

	def add_list(self, lid):
		self.contain_list.add(lid)
		if not self.time_list.has_key(lid):
			self.time_list[lid] = [[]]

class PageGraph(object):
	__slots__ = ('entry', 'nodes', 'edges', 'edges_indexs', 'address_mapping', 'potential_instruction_nodes')
	def __init__(self):
		self.entry = []
		self.nodes = {}
		self.edges = {}
		self.edges_indexs = {}
		self.address_mapping = {}		# Mapping from addr to changed nid
		self.potential_instruction_nodes = {}	#key: node, value: test indexes. Mapping the node and test index in this node
		# self.dot = Digraph(comment='The Page Flow Graph')

	def add_node(self, node):
		if not self.nodes.has_key(node.id):
			self.nodes[node.id] = node

	def add_edge(self, from_node, to_node, file_id):
		# edge = (from_node.id, to_node.id)
		if not self.edges.has_key((from_node, to_node)):
		# if edge not in [(e[0].id, e[1].id)for e in self.edges.keys()]:
			self.edges[(from_node, to_node)] = {file_id: 1}
			self.edges_indexs[(from_node, to_node)] = None
			if to_node.id not in [s.id for s in from_node.successors]:
				from_node.out_degree += 1
				from_node.successors.append(to_node)
			if from_node.id not in [p.id for p in to_node.predcessors]:
				to_node.in_degree += 1
				to_node.predcessors.append(from_node)
		else:
			# count the edge access number for different test cases
			if self.edges[(from_node, to_node)].has_key(file_id):
				self.edges[(from_node, to_node)][file_id] += 1
			else:
				self.edges[(from_node, to_node)][file_id] = 1

		# IPython.embed()
		from_node.order_pattern[to_node.id] = len(from_node.order_pattern) if not from_node.order_pattern.has_key(to_node.id) else from_node.order_pattern[to_node.id]
		try:
			if not from_node.order.has_key(file_id):
				from_node.order[file_id] = [[from_node.order_pattern[to_node.id]]]
			else:
				from_node.order[file_id][-1].append(from_node.order_pattern[to_node.id])
		except AttributeError:
			print '%#x,%#x: %s\t%s' %(from_node.id, to_node.id, from_node.order_pattern, from_node.order)

	def remove_edge(self, from_node, to_node):
		edge = (from_node, to_node)
		if self.edges.has_key(edge):
			self.edges.pop(edge)
		self.nodes[from_node.id].out_degree -= 1
		self.nodes[from_node.id].successors.remove(to_node)
		self.nodes[to_node.id].in_degree -= 1
		self.nodes[to_node.id].predcessors.remove(from_node)

	def add_mapping(self, addr, nid):
		if not self.address_mapping.has_key(addr):
			self.address_mapping[addr] = nid

	def get_node_by_id(self, nid):
		return self.nodes[nid]

	def get_node_by_addr(self, addr):
		for node in self.nodes.values():
			if addr in node.instruction_addrs:
				return node

	# def draw_graph(self):
	# 	self.dot = Digraph(comment='The Page Flow Graph')
	# 	for n in self.nodes.values():
	# 		# self.dot.node('%s' %n.id, '%s, %#x, %s' %(n.id, n.addr, n.contain_list))
	# 		self.dot.node('%s' %n.id, '%#x, %#x' %(n.id, n.bb_addrs))
	# 	for e in self.edges.keys():
	# 		traceid = ''
	# 		for i in e[0].contain_list:
	# 			if i in e[1].contain_list:
	# 				traceid += '%s,' %i
	# 		# self.dot.edge('%s' %e[0].id, '%s' %e[1].id, label=traceid)
	# 		self.dot.edge('%s' %e[0].id, '%s' %e[1].id)

def time_statistical_calc(cfg):
	for n in cfg.nodes.itervalues():
		for f_id, time_lists in n.time_list.iteritems():
			# number of time lists
			l_size = len(time_lists)
			mean_list = []
			std_list = []

			# Get the size of longest list
			max_len = 0
			for t_l in time_lists:
				max_len = len(t_l) if len(t_l) > max_len else max_len

			for i in range(max_len):
				ith_time_list = []
				for j in range(l_size):
					try:
						ith_time_list.append(time_lists[j][i])
					except IndexError:
						continue
				mean_list.append(numpy.mean(ith_time_list))
				std_list.append(numpy.std(ith_time_list))
			n.time_mean[f_id] = mean_list
			n.time_std[f_id] = std_list

def get_pagedcfg_branchlist(branchlist, pdcfg, file_index, size,  angr_cfg=None):
	''' 
	This is a faster method for node CFG creation 
	Two things in this method:
	1. combine the instruction list to build the dcfg for all instruction traces
	2. build dcfg for this instruction trace, and return this dcfg
	'''

	# The first node in il
	# print '\t%s has %s addrs' %(tid, len(il))


	for node in pdcfg.nodes.itervalues():
		if node.time_list.has_key(file_index):
			node.time_list[file_index].append([])
		if node.order.has_key(file_index):
			node.order[file_index].append([])

	bb = branchlist[1]
	pnode = PageNode(bb.start_address/size, bb_addrs=bb.start_address, file_index=file_index)
	pnode.time_list[file_index][-1].append(bb.end_time-bb.start_time)
	pdcfg.add_node(pnode)
	pre_pnode = pnode

	for index, bb in enumerate(branchlist[2:]):
		# Generate Node
		if not pdcfg.nodes.has_key(bb.start_address/size):
			pnode = PageNode(bb.start_address/size, bb_addrs=bb.start_address, file_index=file_index, )
			pnode.time_list[file_index][-1].append(bb.end_time-bb.start_time)
			pdcfg.add_node(pnode)
		else:
			pnode = pdcfg.nodes[bb.start_address/size]
			pnode.bb_addrs.add(bb.start_address)
			pnode.add_list(file_index)

			if pre_pnode.id == pnode.id:
				pnode.time_list[file_index][-1][-1] = pnode.time_list[file_index][-1][-1] + bb.end_time - bb.start_time
			else:
				pnode.time_list[file_index][-1].append(bb.end_time - bb.start_time)

		# Generate Edge
		if pre_pnode.id != pnode.id:
			pdcfg.add_edge(pre_pnode, pnode, file_index)

		pre_pnode = pnode

def major_order(node, fid):
	# Used for order vulnerability detection
	# Get the majority order information.
	orders = {}
	for tid in range(len(node.order[fid])):
		if not orders.has_key(tuple(node.order[fid][tid])):
			orders[tuple(node.order[fid][tid])] = 1
		else:
			orders[tuple(node.order[fid][tid])] += 1
	max_num = 0
	major_orders = None
	for order, num in orders.iteritems():
		if num > max_num:
			max_num = num
			major_orders = order
	return major_orders

def order_vulnerability_detection(cfg):
	# Get the order vulnerable nodes
	order_vulnerable_nodes = []
	for fid, node in cfg.nodes.iteritems():
		# print node
		d = {}
		if len(node.order.keys()) > 0:
			suc_order = major_order(node, node.order.keys()[0])
			for index in node.order.keys()[1:]:
				if suc_order != major_order(node, index):
					order_vulnerable_nodes.append(node)
					break
	return order_vulnerable_nodes

def time_vulnerability_detection(cfg, thresh, option = 'big'):
	time_vulnerable_nodes = []
	for n in cfg.nodes.itervalues():
		vulnerable = False
		for f_id, mean_list in n.time_mean.iteritems():
			if vulnerable is True:
				break
			std = n.time_std[f_id]
			for compare_id, compare_mean_list in n.time_mean.iteritems():
				if vulnerable is True:
					break
				compare_std = n.time_std[compare_id]
				compare_lenth = len(mean_list)
				compare_lenth = len(compare_mean_list) if len(compare_mean_list) < compare_lenth else compare_lenth
				compare_lenth = len(std) if len(std) < compare_lenth else compare_lenth
				for i in range(compare_lenth):
					if option == 'big':
						if (abs(compare_mean_list[i] - mean_list[i]) > abs(compare_std[i] + std[i] + thresh)):
						# if (compare_mean_list[i]-compare_std[i]>mean_list[i]+std[i]+3.0) or (compare_mean_list[i]+compare_std[i] <mean_list[i]-std[i]+3.0):
							vulnerable = True
							time_vulnerable_nodes.append(n)
							break
					else:
						if ( (compare_mean_list[i]>mean_list[i]+std[i]) or (compare_mean_list[i]<mean_list[i]-std[i]) ) and \
						( (mean_list[i]>compare_mean_list[i]+compare_std[i]) or (mean_list[i]<compare_mean_list[i]-compare_std[i])):
							vulnerable = True
							time_vulnerable_nodes.append(n)
	return time_vulnerable_nodes
		
def order_not_vulnerable_time_vulnerable(bCFG, time_vulnerable_nodes, order_vulnerable_nodes):
	order_not_vulnerable_time_vulnerable_nodes_list = set()
	for n in time_vulnerable_nodes:
		if n not in order_vulnerable_nodes:
			order_not_vulnerable_time_vulnerable_nodes_list.add(n)

	return order_not_vulnerable_time_vulnerable_nodes_list

def locator(branchlist, node_addr, num, size):
	'''
	branchlist: trace addresses
	node_addr: 
	num: The num_th out-going edge of the node has vulnerabilty
	size: 4096 / 64/ 1
	'''
	count = -1
	pre_addr = branchlist[1].start_address / size
	for bb in branchlist[2:]:
		if pre_addr/size == node_addr and bb.start_address/size != node_addr:	
			count += 1

		if count == num:
			return [pre_addr, bb.start_address]
		
		pre_addr = bb.start_address

def branch_list_to_branch_dict(branch_list, size):
	'''
	branch_dict:{ (page_node_from_addr, page_node_to_addr): 
															{time_index: (from_address, to_address)}
													
				}
	}
	size: 4096/64/1
	'''
	branch_dict = {}
	pre_bb = branch_list[1]
	for bb in branch_list[2:]:
		if bb.start_address/size != pre_bb.start_address/size:
			if not branch_dict.has_key( pre_bb.start_address/size ):
				branch_dict[pre_bb.start_address/size] = {0: (pre_bb.start_address, bb.start_address)}
			else:
				newid = len(branch_dict[pre_bb.start_address/size])
				branch_dict[pre_bb.start_address/size][newid] = (pre_bb.start_address, bb.start_address)
		pre_bb = bb
	return branch_dict

def time_differetiatable(cfg, page_vuln_nodes, thresh, option='big'):
	results_dict = {}
	diff_ids = set()
	# i = 0
	for node in cfg.nodes.values():
		if node in page_vuln_nodes:
			continue
		# print node
		
		file_ids = node.time_mean.keys()
		mean_len = len(node.time_mean[node.time_mean.keys()[0]])
		for orders in node.order.values():
			for order in orders:
				mean_len = len(order) if len(order) < mean_len else mean_len

		vulnerable = True
		for mean_index in range(mean_len):	# Every accessing time
			for file_id in file_ids:			# between different inputs
				vulnerable = True
				ids = file_ids[:]
				ids.remove(file_id)
				if len(ids) == 0:
					vulnerable = False
					continue
				for file_id_2 in ids:
					if option != 'big':
						if abs(node.time_mean[file_id_2][mean_index]-node.time_mean[file_id][mean_index])<node.time_std[file_id][mean_index]+node.time_std[file_id_2][mean_index]+thresh:
							vulnerable = False
							break
					else:
						if abs(node.time_mean[file_id_2][mean_index]-node.time_mean[file_id][mean_index])<node.time_std[file_id][mean_index]+node.time_std[file_id_2][mean_index]+thresh:
							vulnerable = False
							break
				if vulnerable is True:
					# print node
					# print '%f: %f' %(node.time_mean[file_id][mean_index], node.time_mean[ids[0]][mean_index])
					# print '%f: %f' %(node.time_std[file_id][mean_index], node.time_std[ids[0]][mean_index])
					if results_dict.has_key(node):
						results_dict[node].add(file_id)
					else:
						results_dict[node] = set([file_id])
	print results_dict

def save_results(page_not_vuln_time_vuln_nodes, branch_list, output_path, size, thresh, option = 'big'):
	'''
	result: {page: 
					{(fromaddr, toaddr):
										{mean_index: 
													[(mean, std), (), ... ]}
										}
					}
			}
	'''
	branch_dict = branch_list_to_branch_dict(branch_list, size)
	results_dict = {}
	# i = 0
	for node in page_not_vuln_time_vuln_nodes:
		print node
		# i+= 1
		# print '%d/%d' %(i, len(page_not_vuln_time_vuln_nodes))
		results_dict[node.id] = {}
		file_ids = node.time_mean.keys()
		mean_len = len(node.time_mean[node.time_mean.keys()[0]])
		for orders in node.order.values():
			for order in orders:
				mean_len = len(order) if len(order) < mean_len else mean_len
		# Compare every accessing time of the node between different inputs
		# print mean_len
		# print len(file_ids)
		# i = 0
		vulnerable = False
		for mean_index in range(mean_len):	# Every accessing time
			# if i%100 == 0:
				# print '%d: %s' %(i,time.ctime())
			# i+= 1
			vulnerable = False
			for file_id in file_ids:			# between different inputs
				if vulnerable is True:
					break
				for file_id_2 in file_ids[file_id+1:]:
					if vulnerable is True:
						break
					vulnerable = False
					if option != 'big':
						if (abs(node.time_mean[file_id_2][mean_index] - node.time_mean[file_id][mean_index]) > node.time_std[file_id][mean_index] + node.time_std[file_id_2][mean_index] + thresh):
						# if (node.time_mean[file_id_2][mean_index] > node.time_mean[file_id][mean_index] + node.time_std[file_id][mean_index] or \
						# node.time_mean[file_id_2][mean_index] < node.time_mean[file_id][mean_index]- node.time_std[file_id][mean_index]) and \
						# (node.time_mean[file_id][mean_index] > node.time_mean[file_id_2][mean_index] + node.time_std[file_id_2][mean_index] or \
						# node.time_mean[file_id][mean_index] < node.time_mean[file_id_2][mean_index]- node.time_std[file_id_2][mean_index]):
							vulnerable = True
							# if file_id in l and file_id_2 in l:
							# 	print '%d -> %d, %d: %f:%f; %f:%f' %(mean_index, file_id, file_id_2, node.time_mean[file_id][mean_index], node.time_std[file_id][mean_index], node.time_mean[file_id_2][mean_index], node.time_std[file_id_2][mean_index])
					else:
						if (abs(node.time_mean[file_id_2][mean_index] - node.time_mean[file_id][mean_index]) > node.time_std[file_id][mean_index] + node.time_std[file_id_2][mean_index] + thresh):
						# if (node.time_mean[file_id_2][mean_index] > 2.0 + node.time_mean[file_id][mean_index] + node.time_std[file_id][mean_index] + node.time_std[file_id_2][mean_index] or \
						# node.time_mean[file_id_2][mean_index] < node.time_mean[file_id][mean_index]- 2.0 - node.time_std[file_id][mean_index] - node.time_std[file_id_2][mean_index]) and \
						# (node.time_mean[file_id][mean_index] > 2.0 + node.time_mean[file_id_2][mean_index] + node.time_std[file_id_2][mean_index] + node.time_std[file_id][mean_index]or \
						# node.time_mean[file_id][mean_index] < node.time_mean[file_id_2][mean_index]- 2.0 - node.time_std[file_id_2][mean_index] - node.time_std[file_id][mean_index]):
							vulnerable = True
							# if file_id in l and file_id_2 in l:
							# 	print '%d -> %d, %d: %f:%f; %f:%f' %(mean_index, file_id, file_id_2, node.time_mean[file_id][mean_index], node.time_std[file_id][mean_index], node.time_mean[file_id_2][mean_index], node.time_std[file_id_2][mean_index])
					
					if vulnerable:
						# Get the next page address
						for to_addr, nid in node.order_pattern.iteritems():
							if nid == node.order[file_id][0][mean_index]:
								break
						# Get the basic block level address pair the vulnerability happens
						# addrs = locator(branch_list, node.id, mean_index, 4096)
						if branch_dict.has_key(node.id):
							if branch_dict[node.id].has_key(mean_index):
								addrs = branch_dict[node.id][mean_index]
								# Get the mean time and std which we consider to be vulnerable
								vulnerable_times = []
								for index in file_ids:
									vulnerable_times.append((node.time_mean[index][mean_index], node.time_std[index][mean_index]))

								if addrs is None:
									addrs = [node.id, to_addr]
								if not results_dict[node.id].has_key((addrs[0], addrs[1])):
									results_dict[node.id]['(%#x, %#x)' %(addrs[0], addrs[1])] = {mean_index: vulnerable_times}
								elif not results_dict[node.id][(addrs[0], addrs[1])].has_key(mean_index):
									results_dict[node.id]['(%#x, %#x)' %(addrs[0], addrs[1])][mean_index] = vulnerable_times
						else:
							results_dict[node.id]['(Unknown, Unknown)'] = {}



	with open(output_path+'.' + option+ '.json', 'w') as f:
		json.dump(results_dict, f)
		print 'File %s write finished.' %output_path+'.' + option+ '.json'

	with open(output_path+'.' + option, 'w') as f:
		for page, value in results_dict.iteritems():
			f.write('=====================\n')
			f.write('%#x\n' %page)
			for addrset, timedict in value.iteritems():
				f.write('%s\n' %addrset)
				for index, timelist in timedict.iteritems():
					f.write('%d\n' %index)
					for t in timelist:
						f.write('(%f,%f)\n' %(t[0],t[1]))
				f.write('\n')
		f.write('\n')
		print 'File %s write finished.' %output_path+'.' + option
				
	return results_dict

def save_performance(performance_dict, output_path):
	with open(output_path+'.json', 'w') as f:
		json.dump(performance_dict, f)
		print 'File %s write finished.' %(output_path+'.json')

	with open(output_path, 'w') as f:
		f.write('Number of Inputs: %d\n' %performance_dict['Inputs Number'])
		f.write('Average Inputs Size(MB): %f\n' %performance_dict['Avg Input Size'])
		f.write('Total Analyze Time: %f\n' %performance_dict['Analyze Time'])
		f.write('Num Page Order Vulnerable Nodes: %d\n' %performance_dict['Num Page Order Vulnerable Nodes'])
		f.write('Num Page Time Vulnerable Order NOT Vulnerable Nodes: %d\n' %performance_dict['Num Page Time Vulnerable Order NOT Vulnerable Nodes'])
		f.write('Num Cache Order Vulnerable Nodes: %d\n' %performance_dict['Num Cache Order Vulnerable Nodes'])
		f.write('Num Cache Time Vulnerable Order NOT Vulnerable Nodes: %d\n' %performance_dict['Num Cache Time Vulnerable Order NOT Vulnerable Nodes'])
		f.write('Num Branch Order Vulnerable Nodes: %d\n' %performance_dict['Num Branch Order Vulnerable Nodes'])
		f.write('Num Branch Time Vulnerable Order NOT Vulnerable Nodes: %d\n' %performance_dict['Num Branch Time Vulnerable Order NOT Vulnerable Nodes'])
		print 'File %s write finished.' %output_path

def save_graph(cfg, path):
	cfg_json = {'node':{}, 'edge': {}}
	for node in cfg.nodes.itervalues():
		node_json = {}
		node_json['id'] = node.id
		node_json['bb_addrs'] = list(node.bb_addrs)
		node_json['contain_list'] = list(node.contain_list)
		node_json['out_degree'] = node.out_degree
		node_json['in_degree'] = node.in_degree
		node_json['order_pattern'] = node.order_pattern
		node_json['order'] = node.order
		node_json['time_list'] = node.time_list
		node_json['time_mean'] = node.time_mean
		node_json['time_std'] = node.time_std
		node_json['successors'] = [n.id for n in node.successors]
		node_json['predcessors'] = [n.id for n in node.predcessors]
		cfg_json['node'][node.id] = node_json
	for edge, accesstime in cfg.edges.iteritems():
		cfg_json['edge']['%#x,%#x' %(edge[0].id, edge[1].id)] = accesstime

	with open(path, 'w') as f:
		json.dump(cfg_json, f)
		print 'File %s write finished.' %path

def load_graph(path):
	with open(path, 'r') as f:
		graph_dict = json.load(f)
	pCFG = PageGraph()
	for nid, node in graph_dict['node'].iteritems():
		pnode = PageNode(int(nid))
		pnode.bb_addrs = tuple(node['bb_addrs'])
		pnode.contain_list = tuple(node['contain_list'])
		pnode.out_degree = node['out_degree']
		pnode.in_degree = node['in_degree']
		pnode.order_pattern = node['order_pattern']

		order = {}
		for k,v in node['order'].iteritems():
			order[int(k)] = v
		pnode.order = order

		pnode.time_list = node['time_list']

		time_mean = {}
		for k,v in node['time_mean'].iteritems():
			time_mean[int(k)] = v
		pnode.time_mean = time_mean

		time_std = {}
		for k,v in node['time_std'].iteritems():
			time_std[int(k)] = v
		pnode.time_std = time_std

		pnode.successors = node['successors']
		pnode.predcessors = node['predcessors']
		pCFG.nodes[int(nid)] = pnode

	for edge, accesstime in graph_dict['edge'].iteritems():
		edge = edge.split(',')
		pCFG.edges[(int(edge[0],16),int(edge[1],16))] = accesstime

	return pCFG

def save_order_results(order_vulnerable_nodes, path):
	with open(path, 'w') as f:
		for node in order_vulnerable_nodes:
			f.write('==================\n%#x\n' %node.id)
			for nid, index in node.order_pattern.iteritems():
				f.write('%d: %#x\t' %(index, nid))
			f.write('\n') 
			for fid in node.order.iterkeys():
				f.write('%d:\n%s\n' %(fid, major_order(node, fid)))

def get_vuln_nodes_set(cfg, order_vuln_nodes, time_vuln_nodes):
	nodes_partitions = {}
	for ovn in order_vuln_nodes:
		partitions = {}
		for fid in ovn.order.keys():
			f_order = major_order(ovn, fid)
			if partitions.has_key(tuple(f_order)):
				partitions[tuple(f_order)].append(fid)
			else:
				partitions[tuple(f_order)] = [fid]
		nodes_partitions[ovn] = partitions.values()

	return nodes_partitions


if __name__ == '__main__':
	# Environment Setting
	parser = argparse.ArgumentParser(prog="side-channel-detection.py", description='Detect side-channel vulnerabilities')
	parser.add_argument('-b', dest='bin_path', action='store', required=True, help='The binary to be run')
	parser.add_argument('-i', dest='input_path', action='store',  help='The pt files folder path')
	parser.add_argument('-o', dest='output_path', action='store', help='The path to store output files')
	parser.add_argument('-p', dest='perf_path', action='store', help='The path of the perf program')
	parser.add_argument('-l', dest='load_path', action='store', help='Load the genreated CFGs for detection, and skip the graph generation phase')
	args = parser.parse_args()
	file_name = os.path.basename(args.bin_path)
	if args.output_path is None:
		results_path = './results'
		if not os.path.exists(results_path):
			os.makedirs(results_path)
		args.output_path = os.path.join(results_path, file_name+'_output')
	performance_output_path = os.path.join(args.output_path, file_name+'.performance')
	page_detection_result_output_path = os.path.join(args.output_path, file_name+'.page.result')
	page_order_result_output_path = os.path.join(args.output_path, file_name+'.pageorder.result')
	cache_detection_result_output_path = os.path.join(args.output_path, file_name+'.cache.result')
	cache_order_result_output_path = os.path.join(args.output_path, file_name+'.cacheorder.result')
	branch_detection_result_output_path = os.path.join(args.output_path, file_name+'.branch.result')
	branch_order_result_output_path = os.path.join(args.output_path, file_name+'.branchorder.result')
	if not os.path.exists(args.output_path):
		os.makedirs(args.output_path)

	mapping = None

	# Disable ASLR
	s, echo = commands.getstatusoutput('echo 0 | sudo tee /proc/sys/kernel/randomize_va_space')
	if echo != '0':
		print 'Disable Randomize Failed %s' %echo
	s, echo = commands.getstatusoutput('echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope')
	if echo != '0':
		print 'Disable Ptrace Failed %s' %echo

	# Performance
	performance_dict = {}
	timer = 0
	total_inputs_size = 0

	# DCFG with time info building
	index = 0
	pCFG = PageGraph()
	cCFG = PageGraph()
	bCFG = PageGraph()
	fileIndex = {}
	pt_files = os.walk(args.input_path)
	
	try:
		if args.load_path is None:
			timer = time.time()
			for root, dirs, files in  pt_files:
				if len(files) > 0:
					input_name = os.path.basename(root)
					fileIndex[index] = input_name
					count = 0
					for pt_file_name in files:
						print time.ctime()
						branchlist_addrs = os.path.join(args.output_path, 'perf.branch_list.' + file_name)
						# branchlist_addrs = '/home/wang11488/Projects/SCVScanner/results/perf.branch_list.' + file_name
						print 'sudo %s script -f --ns -i %s > %s' %(args.perf_path, os.path.join(root, pt_file_name),branchlist_addrs)
						cmd = 'sudo %s script -f --ns -i %s > %s' %(args.perf_path, os.path.join(root, pt_file_name),branchlist_addrs)
						os.system(cmd)
						total_inputs_size += os.path.getsize(branchlist_addrs)
						print 'output file size: %fMB' %(os.path.getsize(branchlist_addrs)/float(1024*1024))
						print 'Processing %s' %os.path.join(root, pt_file_name)
						branch_list = ptfileread.preprocess_pt_branch_file(branchlist_addrs, mapping)
						branch_list = ptfileread.filter_library_out(branch_list)
						get_pagedcfg_branchlist(branch_list, pCFG, index, 4096)
						get_pagedcfg_branchlist(branch_list, cCFG, index, 64)
						get_pagedcfg_branchlist(branch_list, bCFG, index, 1)
					
					index += 1
			performance_dict['Total Building Time'] = time.time() - timer
			performance_dict['Inputs Number'] = index
			performance_dict['Avg Input Size'] = total_inputs_size/index/float(1024*1024)


			# Vulnerability Detection
			timer = time.time()
			time_statistical_calc(pCFG)
			page_vuln_nodes = order_vulnerability_detection(pCFG)
			time_vuln_nodes = time_vulnerability_detection(pCFG, 2.0)
			time_vuln_nodes_coarse = time_vulnerability_detection(pCFG, 2.0, option = 'small')
			page_not_vuln_time_vuln_nodes = order_not_vulnerable_time_vulnerable(pCFG, time_vuln_nodes, page_vuln_nodes)
			page_not_vuln_time_vuln_nodes_coarse = order_not_vulnerable_time_vulnerable(pCFG, time_vuln_nodes_coarse, page_vuln_nodes)

			time_statistical_calc(cCFG)
			cache_vuln_nodes = order_vulnerability_detection(cCFG)
			cache_time_vuln_nodes = time_vulnerability_detection(cCFG, 2.0)
			cache_time_vuln_nodes_coarse = time_vulnerability_detection(cCFG, 2.0,  option = 'small')
			cache_not_vuln_time_vuln_nodes = order_not_vulnerable_time_vulnerable(cCFG, cache_time_vuln_nodes, cache_vuln_nodes)
			cache_not_vuln_time_vuln_nodes_coarse = order_not_vulnerable_time_vulnerable(cCFG, cache_time_vuln_nodes_coarse, cache_vuln_nodes)

			time_statistical_calc(bCFG)
			branch_vuln_nodes = order_vulnerability_detection(bCFG)
			branch_time_vuln_nodes = time_vulnerability_detection(bCFG, 2.0)
			branch_time_vuln_nodes_coarse = time_vulnerability_detection(bCFG, 2.0, option = 'small')
			branch_not_vuln_time_vuln_nodes = order_not_vulnerable_time_vulnerable(bCFG, branch_time_vuln_nodes, branch_vuln_nodes)
			branch_not_vuln_time_vuln_nodes_coarse = order_not_vulnerable_time_vulnerable(bCFG, branch_time_vuln_nodes_coarse, branch_vuln_nodes)

			performance_dict['Analyze Time'] = time.time() - timer


			# Results saving
			save_graph(pCFG, os.path.join(args.output_path, file_name+'.pagecfg.json'))
			save_order_results(page_vuln_nodes, page_order_result_output_path)
			results_dict = save_results(page_not_vuln_time_vuln_nodes, branch_list, page_detection_result_output_path, 4096, 2.0)
			performance_dict['Num Page Order Vulnerable Nodes'] = len(page_vuln_nodes)
			performance_dict['Num Page Time Vulnerable Order NOT Vulnerable Nodes'] = len(page_not_vuln_time_vuln_nodes)
			performance_dict['Num Page Time Vulnerable Order NOT Vulnerable Jumps'] = sum([len(v) for v in results_dict.itervalues()])
			

			save_graph(cCFG, os.path.join(args.output_path, file_name+'.cachecfg.json'))
			save_order_results(cache_vuln_nodes, cache_order_result_output_path)
			results_dict = save_results(cache_not_vuln_time_vuln_nodes, branch_list, cache_detection_result_output_path, 64, 2.0)
			# results_dict_coarse = save_results(cache_not_vuln_time_vuln_nodes, branch_list, cache_detection_result_output_path+'.coarse', 64, option = 'small')
			performance_dict['Num Cache Order Vulnerable Nodes'] = len(cache_vuln_nodes)
			performance_dict['Num Cache Time Vulnerable Order NOT Vulnerable Nodes'] = len(cache_not_vuln_time_vuln_nodes)
			performance_dict['Num Cache Time Vulnerable Order NOT Vulnerable Jumps'] = sum([len(v) for v in results_dict.itervalues()])
			
			# save_graph(bCFG, os.path.join(args.output_path, file_name+'.branchcfg.json'))
			# save_order_results(branch_vuln_nodes, branch_order_result_output_path)
			# results_dict = save_results(branch_not_vuln_time_vuln_nodes, branch_list, branch_detection_result_output_path)
			# results_dict_coarse = save_results(branch_not_vuln_time_vuln_nodes, branch_list, branch_detection_result_output_path, option = 'small')
			performance_dict['Num Branch Order Vulnerable Nodes'] = len(branch_vuln_nodes)
			performance_dict['Num Branch Time Vulnerable Order NOT Vulnerable Nodes'] = len(branch_not_vuln_time_vuln_nodes)
			performance_dict['Num Branch Time Vulnerable Order NOT Vulnerable Jumps'] = sum([len(v) for v in results_dict.itervalues()])
			
			save_performance(performance_dict, performance_output_path)

		else:
			print 'Loading PCFG'
			pCFG = load_graph(os.path.join(args.load_path, file_name+'.pagecfg.json'))
			print 'Loading CCFG'
			cCFG = load_graph(os.path.join(args.load_path, file_name+'.cachecfg.json'))
			page_vuln_nodes = order_vulnerability_detection(pCFG)
			time_vuln_nodes = time_vulnerability_detection(pCFG, 2.0)
			page_not_vuln_time_vuln_nodes = order_not_vulnerable_time_vulnerable(pCFG, time_vuln_nodes, page_vuln_nodes)


			cache_vuln_nodes = order_vulnerability_detection(cCFG)
			cache_time_vuln_nodes = time_vulnerability_detection(cCFG, 2.0)
			cache_not_vuln_time_vuln_nodes = order_not_vulnerable_time_vulnerable(cCFG, cache_time_vuln_nodes, cache_vuln_nodes)

	except IndexError:
		print 'IndexError!!!!!'
		exit(1)
	