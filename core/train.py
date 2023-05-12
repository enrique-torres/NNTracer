# general imports
import copy
from pathlib import Path
from math import cos, pi

# PyTorch imports
import torch
from torch.optim.lr_scheduler import StepLR

# Our imports
from core.utils import *
from core.actwgttracer import *
from core.wgtupdatetracer import *
from core.gradtracer import *
from core.sparsitytracer import *

g_arguments = None
g_model = None
g_device = None
g_optimizer = None
g_criterion = None
g_scheduler = None
g_wgtacttracer = None
g_wgtupdtracer = None
g_gradtracer = None
g_sparsitytracer = None
g_exponentstracer = None

def train(epoch, train_loader):
	global g_arguments, g_model, g_device, g_optimizer, g_criterion, g_scheduler, g_wgtacttracer, g_wgtupdtracer, g_gradtracer, g_sparsitytracer, g_exponentstracer

	print('\nEpoch: %2d' % epoch)
	g_model.train()

	losses  = AverageMeter()
	top1    = AverageMeter()
	top5    = AverageMeter()
	stime   = AverageMeter()
	metrics = list()

	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(g_device), target.to(g_device)

		if g_arguments.cosine_lr:
			adjust_learning_rate_cosine(g_optimizer, epoch, batch_idx, len(train_loader))
		
		old_model = None
		if g_arguments.trace_weight_updates:
			#old_model = type(g_model)()
			#old_model.load_state_dict(g_model.state_dict())
			old_model = copy.deepcopy(g_model)
		
		# compute output
		output = g_model(data)       
		loss = g_criterion(output, target)

		prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
		losses.update(loss.item(), data.size(0))
		top1.update(prec1.item(), data.size(0))
		top5.update(prec5.item(), data.size(0))

		# compute gradient and do SGD step
		g_optimizer.zero_grad()
		loss.backward()
		g_optimizer.step()
		# in case a scheduler is used to iterate per batch
		if g_scheduler:
			g_scheduler.step()

		if g_arguments.trace_weight_updates:
			g_wgtupdtracer.calculate_weight_updates(old_model, g_model, g_arguments.model)

		steptime = 0
		steptime = progress_bar(batch_idx, len(train_loader), 'Loss: %2.4f | Top-1: %6.3f%% | Top-5: %6.3f%%' % (losses.avg, top1.avg, top5.avg))
		stime.update(steptime, 1)

		metrics.append([stime.avg, top1.avg, top5.avg, losses.avg])

	write_to_trainlog(g_arguments.output_path, metrics)


def test(test_loader):
	global g_arguments, g_model, g_device, g_optimizer, g_criterion, g_scheduler, g_wgtacttracer, g_wgtupdtracer, g_gradtracer, g_sparsitytracer, g_exponentstracer

	g_model.eval()

	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()    
	metrics = list()

	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(g_device), target.to(g_device)
			output = g_model(data)
			loss = g_criterion(output, target)
			prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))

			losses.update(loss.item(), data.size(0))
			top1.update(prec1.item(), data.size(0))
			top5.update(prec5.item(), data.size(0))

			progress_bar(batch_idx, len(test_loader), 'Loss: %2.4f | Top-1: %6.3f%% | Top-5: %6.3f%% ' % (losses.avg, top1.avg, top5.avg))

	metrics.append([top1.avg, top5.avg, losses.avg])
	write_to_testlog(g_arguments.output_path, metrics)

def train_test_loop(args, model, device, train_loader, val_loader, optimizer, criterion, scheduler, schedule_per_batch=False):
	global g_arguments, g_model, g_device, g_optimizer, g_criterion, g_scheduler, g_wgtacttracer, g_wgtupdtracer, g_gradtracer, g_sparsitytracer, g_exponentstracer

	g_arguments = args
	g_model = model
	g_device = device
	g_optimizer = optimizer
	g_criterion = criterion
	g_scheduler = scheduler
	if g_arguments == None:
		print("No global arguments given to training process, exiting.")
		exit(1)
	elif g_model == None:
		print("No model given to training process, exiting.")
		exit(1)
	elif g_device == None:
		print("No device ID given to training process, exiting.")
		exit(1)
	elif g_optimizer == None:
		print("No optimizer given to training process, exiting.")
		exit(1)
	elif g_criterion == None:
		print("No criterion given to training process, exiting.")
		exit(1)

	if g_arguments.trace_weights or g_arguments.trace_activations:
		g_wgtacttracer = ActWeightTracer(g_model,
										g_arguments.model,
										g_arguments.output_path, 
										g_arguments.trace_weights, 
										g_arguments.trace_activations,
										g_arguments.tracing_start,
										g_arguments.tracing_frequency,
										g_arguments.tracing_limit)
	if g_arguments.trace_gradients:
		g_gradtracer = GradientTracer(g_model, 
										g_arguments.model,
										g_arguments.output_path, 
										g_arguments.tracing_start,
										g_arguments.tracing_frequency,
										g_arguments.tracing_limit)
	if g_arguments.trace_weight_updates:
		g_wgtupdtracer = WeightUpdateTracer(g_model, 
										g_arguments.model,
										g_arguments.output_path, 
										g_arguments.tracing_start,
										g_arguments.tracing_frequency,
										g_arguments.tracing_limit)
	if g_arguments.trace_sparsity:
		g_sparsitytracer = SparsityTracer(g_model, 
										g_arguments.model,
										g_arguments.output_path, 
										g_arguments.tracing_start,
										g_arguments.tracing_frequency,
										g_arguments.tracing_limit)
	if g_arguments.trace_exponents:
		g_exponentstracer = ActWeightExponentsTracer(g_model, 
										g_arguments.model,
										g_arguments.output_path, 
										g_arguments.tracing_start,
										g_arguments.tracing_frequency,
										g_arguments.tracing_limit)

	last_trained_epoch = 0
	if args.load_checkpoint is not None:
		last_trained_epoch, g_model, g_optimizer = load_model_checkpoint(g_arguments, g_model, g_optimizer)
		print("Training from epoch " + str(last_trained_epoch))

	create_logging_folder(g_arguments.output_path)

	for epoch in range(last_trained_epoch, g_arguments.epochs):
		if g_arguments.save_checkpoint:
			save_model_checkpoint(g_arguments, g_model, g_optimizer, epoch)
			
		if not g_arguments.onecycle and not g_arguments.cosine_lr:
			adjust_learning_rate(g_optimizer, epoch, g_arguments)
		else:
			if g_arguments.onecycle:
				print("Last LR value: " + str(g_scheduler.get_last_lr()))
			else:
				print("Last LR value: " + str(g_optimizer.param_groups[0]['lr']))
		# turn on tracing only for training. If tracing for testing is necessary,
		# then remove the lines after train(epoch, train loader) 
		if g_wgtacttracer is not None:
			g_wgtacttracer.set_tracing_state(g_arguments.trace_training)
		if g_gradtracer is not None:
			g_gradtracer.set_tracing_state(g_arguments.trace_training)
		if g_wgtupdtracer is not None:
			g_wgtupdtracer.set_tracing_state(g_arguments.trace_training)
		if g_sparsitytracer is not None:
			g_sparsitytracer.set_tracing_state(g_arguments.trace_training)
		if g_exponentstracer is not None:
			g_exponentstracer.set_tracing_state(g_arguments.trace_training)
		# train for one epoch
		train(epoch, train_loader)

		# turn off tracing for testing, we are not interested
		if g_wgtacttracer is not None:
			g_wgtacttracer.set_tracing_state(g_arguments.trace_testing)
		if g_gradtracer is not None:
			g_gradtracer.set_tracing_state(g_arguments.trace_testing)
		if g_wgtupdtracer is not None:
			g_wgtupdtracer.set_tracing_state(g_arguments.trace_testing)
		if g_sparsitytracer is not None:
			g_sparsitytracer.set_tracing_state(g_arguments.trace_testing)
		if g_exponentstracer is not None:
			g_exponentstracer.set_tracing_state(g_arguments.trace_testing)

		# test on validation set
		test(val_loader)

		if not schedule_per_batch and g_scheduler != None:
			g_scheduler.step()