import math

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from pathlib import Path
from csv import writer
import os
import time

class ActWeightTracerBP():
	def __init__(self, model, network_name, iter_save_list):

		self._module_layer_map = dict()
		self._iter_save_list = None

		self._activation_hook_list = []
		self._activation_exec_map = dict()
		self._activations_output_folder = "act_traces"
		self._weights_output_folder = "weight_traces"

		self._current_bitlength = 23
		self._are_activations_and_weights_being_traced = True

		# Ensure that integer arguments are indeed integers.
		if not isinstance(iter_save_list, list):
			raise ValueError("Iteration saving list must be a list")
		if len(iter_save_list) <= 0
			raise ValueError("Iteration saving list must be of length >= 1")

		# Set the global control variables.
		self._iter_save_list = iter_save_list

		timestr = time.strftime("_%Y%m%d_%H%M%S")

		self._activations_output_folder = network_name + timestr + "/act_traces"
		self._weights_output_folder = network_name + timestr + "/weight_traces"

		# Make a directory to hold these values.
		Path(self._activations_output_folder).mkdir(parents=True, exist_ok=True)
		Path(self._weights_output_folder).mkdir(parents=True, exist_ok=True)

		self._hook_activations("net", model)

		# Save the model layout and the mapping to a file.
		with open(Path("./" + network_name + timestr + "/modelmap.log"), "w+") as f:
			f.write(str(model)+"\n=================================\n\n")
			for module, mapping in self._module_layer_map.items():
				f.write(str(module)+":"+str(mapping)+"\n\n")
		print("Created hooks to trace activations and weights")

	def _hook_activations(self, top_name, model):
		layer_index = 0
		for name, module in model.named_children():
			type(module)
			if module == model:
				continue
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, Layered_MatMul):
				if module in self._module_layer_map:
					print("Error.")
					exit(-1)
				self._module_layer_map[module] = top_name + "_" + name
				layer_index += 1
				self._activation_hook_list.append(module.register_forward_hook(self._activation_hook_fn))
			else:
				self._hook_activations(top_name + "_" + name, module)

	def set_tracing_state(self, state: bool):
		self._are_activations_and_weights_being_traced = state
	
	def _activation_hook_fn(self, module, inputs, outputs):

		#print("Hook is happening")
		if not self._are_activations_and_weights_being_traced:
			return

		if self._iter_save_list is None:
			raise ValueError(
				"_activation_hook_fn has non-initialized runtime parameters.")

		# Fetch the module_name
		module_name = self._module_layer_map[module]
		# Check if this is in our data structure
		# that holds reference to the number of times forward propagation occured
		# for this module. If not, install an entry and set the count to 0.
		if module_name not in self._activation_exec_map:
			self._activation_exec_map[module_name] = 0

		# Grab the current number of times this module has undergone fp and how many
		# times we've saved the result
		current_bpiter = self._activation_exec_map[module_name]

		# Check if the current iteration is in the list of iterations we want to store
		if current_bpiter not in self._iter_save_list:
			# If it is not, we don't need to capture, so increment iteration and exit.
			self._activation_exec_map[module_name] += 1
			return

		activations_prefix = self._activations_output_folder + "/" + module_name + "_id" + str(current_fp_iter)
		weights_prefix = self._weights_output_folder + "/" + module_name + "_id" + str(current_fp_iter) + "_"

		state = module.state_dict()
		for i in state:
			# Conditionally select the state only if "weight" is in the parameter str.
			if "weight" in i:
				np.save(weights_prefix + i, state[i].cpu())
			if "alpha_w" in i:
				np.save(weights_prefix + i + "_alpha_w", state[i].cpu())
			if "alpha_a" in i:
				np.save(weights_prefix + i + "_alpha_a", state[i].cpu())
		
		np.save(activations_prefix, outputs.clone().detach().cpu())

		# Update the number of times this module has undergone forward propagation.
		self._activation_exec_map[module_name] += 1