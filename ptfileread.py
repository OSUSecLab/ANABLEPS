import sys
# from elftools.elf.elffile import ELFFile
# import capstone
# from capstone.x86 import *
# import IPython
import sys
import copy
import time


class Block:
	start_address = None
	end_address = None
	time = None			# avg time of this node
	time_list = None	# time list of this node
	time_sum = None
	next_block = None	# dict
	prev_block = None	# dict
	order = None	#list

def preprocess_pt_branch_file(filename, mapping):

	class BranchBlock:
		start_address = None
		end_address = None
		start_time = None
		end_time = None
		func_name = None
		cycles = 0

	f = open(filename, 'r')
	branch_list = []	# The list for all branch information
	prev_time = None
	block_list = []		# The temporary list for branches with the same time
	i = 0
	for l in f:
		items = l.split(' ')
		while True:
			try:
				items.remove('')
			except ValueError:
				break
		# filter out error msgs
		if len(items) != 13:
			continue
		block_end_addr = int(items[6], 16) if int(items[6], 16) != 0 else 0xffffffff
		func_name = items[7]
		next_block_start_addr = int(items[10], 16)	if int(items[10], 16) != 0 else 0xffffffff
		time = int(float(items[3][:-1]) * 1000000000)

		# print '%s: %#x => %#x' %(time, block_end_addr, next_block_start_addr)

		# block = BranchBlock() if len(branch_list) == 0 else branch_list[-1]
		if len(block_list) != 0:
			block = block_list[-1]
		else:
			if len(branch_list) != 0:
				block = branch_list[-1]
			else:
				block = BranchBlock()
		block.end_address = block_end_addr
		block.func_name = func_name

		next_block = BranchBlock()
		next_block.start_address = next_block_start_addr

		if prev_time is None:
			prev_time = time
			branch_list.append(block)
			block_list.append(next_block)
			continue

		if time == prev_time:
			block_list.append(next_block)
		else:
			# Recalculate time for each block in block_list
			if len(block_list) > 1:
				start_address = branch_list[-1].end_address
				end_address = block_end_addr

				total_time = time - prev_time
				# Calculate the start and end time for each block
				start_time = prev_time
				end_time = None
				time_gap = float(total_time)/float(len(block_list))
				for bb in block_list:
					end_time = start_time + time_gap
					bb.start_time = start_time
					bb.end_time = end_time
					start_time = end_time

				# Merge the block list into branch list
				for bb in block_list:
					branch_list.append(bb)
				# branch_list = branch_list + block_list
				block_list = [next_block]

			else:
				block_list[0].start_time = prev_time
				block_list[0].end_time = time
				for bb in block_list:
					branch_list.append(bb)
				block_list = [next_block]

			prev_time = time

	# Change the address based on mapping
	if mapping is not None:
		for bb in branch_list:
			for orig_addrs in mapping.iterkeys():
				if orig_addrs[0] <= bb.start_address < orig_addrs[1]:
					bb.start_address = mapping[orig_addrs] + bb.start_address - orig_addrs[0]
				if orig_addrs[0] <= bb.end_address < orig_addrs[1]:
					bb.end_address = mapping[orig_addrs] + bb.end_address - orig_addrs[0]

	return branch_list

def filter_library_out(branch_list):
	new_branch_list = []
	in_lib = False
	in_lib_bbs = []
	for bb in branch_list:
		if bb.start_address < 0xf000000:
			if in_lib is True:
				new_bb = copy.deepcopy(in_lib_bbs[0])
				new_bb.end_address = in_lib_bbs[-1].end_address
				new_bb.end_time = in_lib_bbs[-1].end_time
				new_bb.cycles = sum([lib_bb.cycles for lib_bb in in_lib_bbs])
				new_branch_list.append(new_bb)

			new_branch_list.append(bb)
			in_lib = False
			in_lib_bbs = []
		else:
			in_lib = True
			in_lib_bbs.append(bb)

	return new_branch_list

def pt_branch_list_to_branch_dict(branch_list, disasm_cycle=None):

	def init_block(block, branch):
		block.start_address = branch.start_address
		block.end_address = branch.end_address
		# print '%#x,%#x, %s' %(branch.start_address, branch.end_address, branch.end_time-branch.start_time)
		if branch.end_time is not None and branch.start_time is not None:	
			block.time_list = [branch.end_time - branch.start_time] if block.time_list is None else block.time_list.append(branch.end_time - branch.start_time)
			block.time_sum = branch.end_time - branch.start_time if block.time_sum is None else block.time_sum + branch.end_time - branch.start_time
			block.time = block.time_sum / len(block.time_list)
		
		block.next_block = {}
		block.prev_block = {}
		block.order = []
		return block

	def add_time(block, branch):
		block.time_list.append(branch.end_time - branch.start_time)
		block.time_sum = block.time_sum + branch.end_time - branch.start_time
		block.time = block.time_sum / len(block.time_list)
		

	branch_dict = {}
	prev_block = None
	next_block = None
	size = len(branch_list)
	print 'list size: %d' %size
	for i, bb in enumerate(branch_list):

		if branch_dict.has_key(bb.start_address):
			block = branch_dict[bb.start_address]
			add_time(block, branch_list[i])
		else:
			block = init_block(Block(),branch_list[i])
			branch_dict[block.start_address] = block
			
		if prev_block is not None:
			block.prev_block[prev_block.start_address] = len(block.prev_block) if not block.prev_block.has_key(prev_block.start_address) else block.prev_block[prev_block.start_address]
			prev_block.next_block[block.start_address] = len(prev_block.next_block) if not prev_block.next_block.has_key(block.start_address) else prev_block.next_block[block.start_address]
			prev_block.order.append(prev_block.next_block[block.start_address])

		prev_block = block
		
	return branch_dict

def get_cycle(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
		cycles = {}
		for line in lines:
			cmd = line.split('\t')
			opcode = cmd[0]
			operand = cmd[1]
			throughput = float(cmd[-1].strip())
			operand = operand.split(',')
			operand_dic = {} if not cycles.has_key(opcode.lower()) else cycles[opcode.lower()]
			operand_tmp = []
			for op in operand:
				if op == 'r':
					operand_tmp.append(X86_OP_REG)
				if op == 'i':
					operand_tmp.append(X86_OP_IMM)
				if op == 'm':
					operand_tmp.append(X86_OP_MEM)
			operand = tuple(operand_tmp)
			operand_dic[operand] = throughput
			cycles[opcode.lower()] = operand_dic
	return cycles

# def get_disasm_cycle(filename, cycles):
# 	with open(filename, 'rb') as f:
# 		elffile = ELFFile(f)
# 		for seg in elffile.iter_segments():
# 			if seg.header['p_flags'] == 5 and seg.header['p_type'] == 'PT_LOAD':	# Executable load seg
# 				bytes = seg.data()
# 				break
# 		list_sec = []
# 		for section in elffile.iter_sections():
# 			d = {}
# 			if section.name == '.init':
# 				d['name'] = '.init'
# 			elif section.name == '.plt':
# 				d['name'] = '.plt'
# 			elif section.name == '.text':
# 				d['name'] = '.text'
# 			elif section.name == '.fini':
# 				d['name'] = '.fini'
# 			else:
# 				continue
# 			d['offs'] = section.header.sh_offset
# 			d['size'] = section.header.sh_size
# 			d['addr'] = section.header.sh_addr
# 			list_sec.append(d)

# 	insn_dic = {}
# 	md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
# 	md.detail = True
# 	# IPython.embed()
# 	for section in list_sec:
# 		for i in md.disasm(bytes[section['offs']:section['offs']+section['size']], section['addr']):
# 			ins = {'opcode':i.mnemonic, 'operand':i.operands}
# 			operand = []
# 			for op in ins['operand']:
# 				operand.append(op.type)
# 			if not cycles.has_key(ins['opcode'].lower()):
# 				cycle = 1
# 			else:
# 				# The cycles obtained is not accurate due to the in-completeness of cpu-cycle-file
# 				cycle = cycles[ins['opcode'].lower()][tuple(operand)] if cycles[ins['opcode'].lower()].has_key(tuple(operand)) else cycles[ins['opcode'].lower()].values()[0]
# 			insn_dic[i.address] = cycle
# 	return insn_dic

def retrieve_addr_mapping(filepath):
	with open(filepath, 'r') as f:
		mapping = f.readlines()
	addr_mapping = {}
	for map_item in mapping:
		items = map_item.strip().split('\t')
		lib_name = items[0]
		orig_addrs = (int(items[1],16), int(items[2],16))
		mapped_addrs = (int(items[3],16), int(items[4],16))
		addr_mapping[orig_addrs] = mapped_addrs
	return addr_mapping

if __name__ == '__main__':
	# python ptfileread.py binary pt-branch-file cpu-cycle-file
	# binary = sys.argv[1]
	filename = sys.argv[1]
	# cyclefile = sys.argv[3]
	# mapping_file_path = sys.argv[2]
	# ins_cycles = get_cycle(cyclefile)
	# disasm_cycle = get_disasm_cycle(binary, ins_cycles)
	# branch_list = preprocess_pt_branch_file(filename, disasm_cycle)
	# print time.ctime()
	# mapping = retrieve_addr_mapping(mapping_file_path)
	mapping = None
	branch_list = preprocess_pt_branch_file(filename,mapping)
	branch_list_without_libs = filter_library_out(branch_list)
	branch_dict = pt_branch_list_to_branch_dict(branch_list_without_libs)
	# print time.ctime()
	IPython.embed()