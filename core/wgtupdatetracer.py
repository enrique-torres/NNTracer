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

class WeightUpdateTracer():
	def __init__(self, model, network_name, output_path, start_save_at=0, save_every_ith=10, capture_maximum=10):

		self._module_layer_map = dict()

		self._save_weight_updates_every_ith = None
		self._weight_updates_capture_maximum = None
		self._weight_updates_start_save_at = None

		self._network_name = network_name

		self._weight_updates_output_folder = "weight_updates_traces"

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
		self._weight_updates_start_save_at = start_save_at
		self._save_weight_updates_every_ith = save_every_ith
		self._weight_updates_capture_maximum = capture_maximum
		self._are_weight_updates_being_traced = True

		self._current_bpiter = 0
		self._current_num_traces = 0

		self._weight_updates_output_folder = output_path + "/weight_updates_traces"

		Path(self._weight_updates_output_folder).mkdir(parents=True, exist_ok=True)

		self._read_model_structure(self._network_name, model)

		# Save the model layout and the mapping to a file.
		with open(output_path + "/modelmap.log", "w+") as f:
			f.write(str(model)+"\n=================================\n\n")
			for module, mapping in self._module_layer_map.items():
				f.write(str(module)+":"+str(mapping)+"\n\n")

		print("Will do weight update tracing")

	def _read_model_structure(self, top_name, model):
		for name, module in model.named_children():
			if module == model:
				continue
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.ReLU) or isinstance(module, nn.BatchNorm2d):
				if module in self._module_layer_map:
					print("Error.")
					exit(-1)
				self._module_layer_map[module] = top_name + "_" + name
			else:
				self._read_model_structure(top_name + "_" + name, module)

	def set_tracing_state(self, state: bool):
		self._are_weight_updates_being_traced = state
	
	def calculate_weight_updates(self, model_before_update, model_after_update, top_name):
		if not self._are_weight_updates_being_traced:
			return
		if self._current_bpiter < self._weight_updates_start_save_at or self._current_num_traces >= (self._weight_updates_capture_maximum):
			return
		if self._current_bpiter % self._save_weight_updates_every_ith != 0:
			return

		if self._are_weight_updates_being_traced:
			model_after_update_children = {name:module for name,module in model_after_update.named_children()}
			for name, module in model_before_update.named_children():
				if module == model_before_update:
					continue
				if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.ReLU) or isinstance(module, nn.BatchNorm2d):
					state = module.state_dict()
					for i in state:
						# Conditionally select the state only if "weight" is in the parameter str.
						if "weight" in i:
							after_update_state_dict = model_after_update_children[name].state_dict()
							module_weight_update = state[i].cpu() - after_update_state_dict[i].cpu()
							np.save(self._weight_updates_output_folder + "/" + top_name + "_" + name + "_weightupd", module_weight_update)
				else:
					self._recursive_calculate_weight_updates(module, model_after_update_children[name], top_name + "_" + name)

			self._current_bpiter += 1
	
	def _recursive_calculate_weight_updates(self, module_before_update, module_after_update, top_name):
		module_after_update_children = {name:module for name,module in module_after_update.named_children()}
		for name, module in module_before_update.named_children():
				if module == module_before_update:
					continue
				if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.ReLU) or isinstance(module, nn.BatchNorm2d):
					state = module.state_dict()
					for i in state:
						# Conditionally select the state only if "weight" is in the parameter str.
						if "weight" in i:
							after_update_state_dict = module_after_update_children[name].state_dict()
							module_weight_update = state[i].cpu() - after_update_state_dict[i].cpu()
							np.save(self._weight_updates_output_folder + "/" + top_name + "_" + name + "_weightupd", module_weight_update)
				else:
					self._recursive_calculate_weight_updates(module, module_after_update_children[name], top_name + "_" + name)