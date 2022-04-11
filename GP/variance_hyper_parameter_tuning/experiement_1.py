import tensorflow as tf
# tf.config.experimental.list_physical_devices('GPU')
if tf.test.gpu_device_name():
    print('\n','\n','hello')
    print('Deafault: {}'.format(tf.test.gpu_device_name()))
else:
    print("please install GPU")
# import tensorflow as tf
# from tensorboard.plugins.hparams import api as hp
# import gpflow
# import numpy as np

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# from data_loader import loader
# from helper_functions import helpers, gpflow_new_kernels
# import matplotlib.pyplot as plt
# import gpflow_new_kernels_modifed_1
# # %load_ext tensorboard



# #  data loaded  as x_train,y_train,x_test,y_test
# np.random.seed(42)
# tf.random.set_seed(42)



# def pre_process_data(xtrain, ytrain, xtest, ytest, mapToProbability=False, standardScale=False, jitter=False, asFloat64=False):
#     np.random.seed(42)
#     tf.random.set_seed(42)
#     if jitter:
#         xtest, xtrain = xtest + np.random.uniform(low=1e-7, high=1e-5, size=xtest.shape), xtrain + np.random.uniform(low=1e-7, high=1e-5, size=xtrain.shape)
#     if mapToProbability:
#         xtrain, xtest = helpers.map_to_probability_simplex([xtrain, xtest])
#     if standardScale:
#         scaler = StandardScaler()
#         scaler.fit(xtrain)
#         xtrain, xtest = scaler.transform(xtrain), scaler.transform(xtest)
#     if asFloat64:
#         xtrain, ytrain, xtest, ytest = list(map(lambda x: x.astype(np.float64), [xtrain, ytrain, xtest, ytest]))

#     return xtrain, ytrain, xtest, ytest


# # MNIST DATA
# nsx, nsy = 28, 28
# m = helpers.get_image_cost_matrix(nsx, nsy)
# num_training_per_class = [5]
# all_accuracies = []
# num_test_per_class = 2

# x_train, y_train, x_test, y_test = loader.data_loader(dataset="mnist", dataset_type="partial",
#                                           num_each_class_train=1, num_each_class_test=num_test_per_class,
#                                           seed=42)
# x_train, y_train, x_test, y_test = pre_process_data(xtrain=x_train.copy(), ytrain=y_train.copy(), xtest=x_test, ytest=y_test,
#                                         standardScale=False, mapToProbability=True,
#                                         jitter=True, asFloat64=True)

# precomputed_test_test_wassertein_distances = helpers.get_wassertein_distances(
#     cost_matrix=m, x1=x_test, label='Pre-computing Test Test Wassertein Distances')
# precomputed_train_train_wassertein_distances = helpers.get_wassertein_distances(
#                  cost_matrix=m, x1=x_train, label='Pre-computing Train Train Wassertein Distances')
# precomputed_train_test_wassertein_distances = helpers.get_wassertein_distances(
#                  cost_matrix=m, x1=x_test,x2 = x_train, label='Pre-computing Train Test Wassertein Distances')

# # 

# HP_VARIANCE = hp.HParam('variance',hp.RealInterval(0.1,50.0))

# METRIC_ACCURACY = 'likelihood'

# with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
#   hp.hparams_config(
#     hparams=[HP_VARIANCE],
#     metrics=[hp.Metric(METRIC_ACCURACY, display_name='likelihood')],
#   )

# def train_model(hparams):
#     C = 10
#     model = gpflow.models.VGP(data=(x_train, y_train),
#                           mean_function=gpflow.mean_functions.Zero(),
#                           kernel=gpflow_new_kernels_modifed_1.ExpWassWithPsdApproximation(
#                               cost_matrix=m,initial_variance= hparams[HP_VARIANCE], initial_scale=1.0, reg=0.4, stopThr=5e-2,
#                               numIterMax=1000, xtrain=x_train, xtest=x_test,
#                               precomputed_train_train_wassertein_distances = precomputed_train_train_wassertein_distances,
#                               precomputed_train_test_wassertein_distances = precomputed_train_test_wassertein_distances,
#                                           precomputed_test_test_wassertein_distances = precomputed_test_test_wassertein_distances,y_train = y_train,Maxiters=12),
#                           likelihood=gpflow.likelihoods.MultiClass(num_classes=C,
#                                                                    invlink=gpflow.likelihoods.RobustMax(
#                                                                        num_classes=C)),
#                           num_latent_gps=C)
    
        
#     # set trainable parameters
#     gpflow.utilities.set_trainable(model.kernel.variance, True)
#     gpflow.utilities.set_trainable(model.kernel.scale, True)
#     gpflow.utilities.set_trainable(model.likelihood.invlink.epsilon, False)
#     model.kernel.mode = "training"

#     # see model parameters after training
#     gpflow.utilities.print_summary(model)

#     # running optimization
#     opt = gpflow.optimizers.Scipy()
#     opt_logs = opt.minimize(
#         model.training_loss_closure(compile=False),
#         model.trainable_variables,
#         compile=False
#     )
    
#     return model. maximum_log_likelihood_objective()
    
    
# def run(run_dir, hparams):
#     with tf.summary.create_file_writer(run_dir).as_default():
#         hp.hparams(hparams)  # record the values used in this trial
#         accuracy = train_model(hparams)
#         tf.summary.scalar(METRIC_ACCURACY, accuracy, step=2)
    

# session_num = 0

# for var in (HP_VARIANCE.domain.min_value,HP_VARIANCE.domain.max_value):
#         hparams = {
#             HP_VARIANCE: var
#         }
#         run_name = "run-%d" % session_num
#         print('--- Starting trial: %s' % run_name)
#         print({h.name: hparams[h] for h in hparams})
#         run('logs/hparam_tuning/' + run_name, hparams)
#         session_num += 1

