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

class ActWeightTracer():
	def __init__(self, model, output_path, trace_weights, trace_activations, start_save_at=0, save_every_ith=10, capture_maximum=10):

		self._module_layer_map = dict()
		self._save_activation_every_ith = None
		self._activation_capture_maximum = None
		self._activation_start_save_at = None

		self._activation_hook_list = []
		self._activation_exec_map = dict()
		self._activation_save_map = dict()
		self._activations_output_folder = "act_traces"
		self._weights_output_folder = "weight_traces"

		# Ensure that integer arguments are indeed integers.
		if not isinstance(start_save_at, int):
			raise ValueError("start_save_at must be an integer.")

		if not isinstance(save_every_ith, int):
			raise ValueError("save_every_ith must be an integer.")

		if not isinstance(capture_maximum, int):
			raise ValueError("capture_maximum must be an integer.")

		if start_save_at < 0:
			start_save_at = 0

		if save_every_ith <= 0:
			save_every_ith = 1

		if capture_maximum < 0:
			capture_maximum = 1

		# Set the global control variables.
		self._activation_start_save_at = start_save_at
		self._save_activation_every_ith = save_every_ith
		self._activation_capture_maximum = capture_maximum
		self._are_activations_and_weights_being_traced = True
		self._trace_weights_active = trace_weights
		self._trace_activations_active = trace_activations


		#timestr = time.strftime("_%Y%m%d_%H%M%S")

		self._activations_output_folder = output_path + "/act_traces"
		self._weights_output_folder = output_path + "/weight_traces"

		# Make a directory to hold these values.
		Path(self._activations_output_folder).mkdir(parents=True, exist_ok=True)
		Path(self._weights_output_folder).mkdir(parents=True, exist_ok=True)

		self._hook_activations("net", model)

		# Save the model layout and the mapping to a file.
		with open(Path("./" + output_path + "/modelmap.log"), "w+") as f:
			f.write(str(model)+"\n=================================\n\n")
			for module, mapping in self._module_layer_map.items():
				f.write(str(module)+":"+str(mapping)+"\n\n")
		print("Created hooks to trace activations and weights")

	def _hook_activations(self, top_name, model):
		for name, module in model.named_children():
			if module == model:
				continue
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.ReLU) or isinstance(module, nn.BatchNorm2d):
				if module in self._module_layer_map:
					print("Error.")
					exit(-1)
				self._module_layer_map[module] = top_name + "_" + name
				self._activation_hook_list.append(module.register_forward_hook(self._activation_hook_fn))
			else:
				self._hook_activations(top_name + "_" + name, module)

	def set_tracing_state(self, state: bool):
		self._are_activations_and_weights_being_traced = state
	
	def _activation_hook_fn(self, module, inputs, outputs):

		#print("Hook is happening")
		if not self._are_activations_and_weights_being_traced:
			return

		for param in [self._activation_capture_maximum, self._save_activation_every_ith, self._activation_start_save_at]:
			if param is None:
				raise ValueError(
					"_activation_hook_fn has non-initialized runtime parameters.")

		# Fetch the module_name
		module_name = self._module_layer_map[module]
		# Check if this is in our data structure
		# that holds reference to the number of times forward propagation occured
		# for this module. If not, install an entry and set the count to 0.
		if module_name not in self._activation_exec_map:
			self._activation_exec_map[module_name] = 0

		# Check if this is in our data structure
		# that holds reference to the number of times we've saved the result of the forward propagation
		# for this module. If not, install an entry and set the count to 0.
		if module_name not in self._activation_save_map:
			self._activation_save_map[module_name] = 0

		# Grab the current number of times this module has undergone fp and how many
		# times we've saved the result
		current_bpiter = self._activation_exec_map[module_name]
		current_num_saves = self._activation_save_map[module_name]


		# Check if the current number of times we've undergone fp is less than when we wanted to start saving.
		# or if we've saved enough runs (as per request)
		if current_bpiter < self._activation_start_save_at or current_num_saves >= (self._activation_capture_maximum):
			# If so, We've captured all we've wanted to, so exit.
			self._activation_exec_map[module_name] += 1
			return

		# Now check if it's time for us to save the activation.
		if self._activation_exec_map[module_name] % self._save_activation_every_ith != 0:
			self._activation_exec_map[module_name] += 1
			return

		activations_prefix = self._activations_output_folder + "/" + module_name + "_iter" + str(current_bpiter)
		weights_prefix = self._weights_output_folder + "/" + module_name + "_iter" + str(current_bpiter) + "_"

		if self._trace_weights_active:
			state = module.state_dict()
			for i in state:
				# Conditionally select the state only if "weight" is in the parameter str.
				if "weight" in i:
					np.save(weights_prefix + i, state[i].cpu())
		
		if self._trace_activations_active:
			np.save(activations_prefix, outputs.clone().detach().cpu())

		# Update the number of times this module has undergone forward propagation.
		self._activation_exec_map[module_name] += 1
		self._activation_save_map[module_name] += 1