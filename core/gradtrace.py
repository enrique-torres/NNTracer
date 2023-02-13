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

class GradientTrace():
	def __init__(self, model, network_name, time_string, start_save_at=0, save_every_ith=10, capture_maximum=10):
		
		self._module_layer_map = dict()
		self._save_gradients_every_ith = None
		self._gradients_capture_maximum = None
		self._gradients_start_save_at = None

		self._gradients_hook_list = []
		self._gradients_exec_map = dict()
		self._gradients_save_map = dict()
		self._gradients_output_folder = "gradient_traces"

		self._are_gradients_being_traced = True

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
		self._gradients_start_save_at = start_save_at
		self._save_gradients_every_ith = save_every_ith
		self._gradients_capture_maximum = capture_maximum

		#timestr = time.strftime("_%Y%m%d_%H%M%S")

		self._gradients_output_folder = network_name + time_string + "/gradient_traces"

		Path(self._gradients_output_folder).mkdir(parents=True, exist_ok=True)

		self._hook_gradients("net", model)
		print("Created hooks to trace gradients!")
		

	def _hook_gradients(self, top_name, model):
		layer_index = 0
		for name, module in model.named_children():
			type(module)
			if module == model:
				continue
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.ReLU) or isinstance(module, nn.BatchNorm2d):
				if module in self._module_layer_map:
					print("Error.")
					exit(-1)
				self._module_layer_map[module] = top_name + "_" + name
				layer_index += 1
				self._gradients_hook_list.append(module.register_forward_hook(self._gradients_hook_fn))
			else:
				self._hook_gradients(top_name + "_" + name, module)

	def set_tracing_state(self, state: bool):
		self._are_gradients_being_traced = state

	def _gradients_hook_fn(self, module, grad_input, grad_output):
		if not self._are_gradients_being_traced:
			return

		for param in [self._gradients_capture_maximum, self._save_gradients_every_ith, self._gradients_start_save_at]:
			if param is None:
				raise ValueError(
					"_gradients_hook_fn has non-initialized runtime parameters.")

		# Fetch the module_name
		module_name = self._module_layer_map[module]
		# Check if this is in our data structure
		# that holds reference to the number of times forward propagation occured
		# for this module. If not, install an entry and set the count to 0.
		if module_name not in self._gradients_exec_map:
			self._gradients_exec_map[module_name] = 0

		# Check if this is in our data structure
		# that holds reference to the number of times we've saved the result of the forward propagation
		# for this module. If not, install an entry and set the count to 0.
		if module_name not in self._gradients_save_map:
			self._gradients_save_map[module_name] = 0

		# Grab the current number of times this module has undergone fp and how many
		# times we've saved the result
		current_bpiter = self._gradients_exec_map[module_name]
		current_num_saves = self._gradients_save_map[module_name]


		# Check if the current number of times we've undergone fp is less than when we wanted to start saving.
		# or if we've saved enough runs (as per request)
		if current_bpiter < self._gradients_start_save_at or current_num_saves >= (self._gradients_capture_maximum):
			# If so, We've captured all we've wanted to, so exit.
			self._gradients_exec_map[module_name] += 1
			return

		# Now check if it's time for us to save the gradients.
		if self._gradients_exec_map[module_name] % self._save_gradients_every_ith != 0:
			self._gradients_exec_map[module_name] += 1
			return

		input_gradients_prefix = self._gradients_output_folder + "/" + module_name + "_ingrads_iter" + str(current_bpiter)
		output_gradients_prefix = self._gradients_output_folder + "/" + module_name + "_outgrads_iter" + str(current_bpiter)

		np.save(input_gradients_prefix, grad_input.clone().detach().cpu())
		np.save(output_gradients_prefix, grad_output.clone().detach().cpu())


		# Update the number of times this module has undergone forward propagation.
		self._gradients_exec_map[module_name] += 1
		self._gradients_save_map[module_name] += 1