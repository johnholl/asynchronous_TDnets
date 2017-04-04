#!/usr/bin/env python
import argparse
import distutils.version
import logging
import os
import signal
import sys
import time
import tensorflow as tf
from lab_interface import LabInterface
from agents.a3c import A3C
from agents.a3c_pc import A3CPC
from agents.a3c_fc import A3CFC
from agents.a3c_pc_r import A3CPCR
from agents.basic_tdnet_simplified import ATDNet
from agents.conv_value_tdnet import ConvTDAgent


from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops

from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary as _summary
from tensorflow.python.training import coordinator
from tensorflow.python.training import session_manager as session_manager_mod

USE_DEFAULT = 0

class NonFinalSupervisor(tf.train.Supervisor):
  def __init__(self,
               graph=None,
               ready_op=USE_DEFAULT,
               ready_for_local_init_op=USE_DEFAULT,
               is_chief=True,
               init_op=USE_DEFAULT,
               init_feed_dict=None,
               local_init_op=USE_DEFAULT,
               logdir=None,
               summary_op=USE_DEFAULT,
               saver=USE_DEFAULT,
               global_step=USE_DEFAULT,
               save_summaries_secs=120,
               save_model_secs=600,
               recovery_wait_secs=30,
               stop_grace_secs=120,
               checkpoint_basename="model.ckpt",
               session_manager=None,
               summary_writer=USE_DEFAULT,
               init_fn=None):
    """Create a `Supervisor`.

    Args:
      graph: A `Graph`.  The graph that the model will use.  Defaults to the
        default `Graph`.  The supervisor may add operations to the graph before
        creating a session, but the graph should not be modified by the caller
        after passing it to the supervisor.
      ready_op: 1-D string `Tensor`.  This tensor is evaluated by supervisors in
        `prepare_or_wait_for_session()` to check if the model is ready to use.
        The model is considered ready if it returns an empty array.  Defaults to
        the tensor returned from `tf.report_uninitialized_variables()`  If
        `None`, the model is not checked for readiness.
      ready_for_local_init_op: 1-D string `Tensor`.  This tensor is evaluated by
        supervisors in `prepare_or_wait_for_session()` to check if the model is
        ready to run the local_init_op.
        The model is considered ready if it returns an empty array.  Defaults to
        the tensor returned from
        `tf.report_uninitialized_variables(tf.global_variables())`. If `None`,
        the model is not checked for readiness before running local_init_op.
      is_chief: If True, create a chief supervisor in charge of initializing
        and restoring the model.  If False, create a supervisor that relies
        on a chief supervisor for inits and restore.
      init_op: `Operation`.  Used by chief supervisors to initialize the model
        when it can not be recovered.  Defaults to an `Operation` that
        initializes all variables.  If `None`, no initialization is done
        automatically unless you pass a value for `init_fn`, see below.
      init_feed_dict: A dictionary that maps `Tensor` objects to feed values.
        This feed dictionary will be used when `init_op` is evaluated.
      local_init_op: `Operation`. Used by all supervisors to run initializations
        that should run for every new supervisor instance. By default these
        are table initializers and initializers for local variables.
        If `None`, no further per supervisor-instance initialization is
        done automatically.
      logdir: A string.  Optional path to a directory where to checkpoint the
        model and log events for the visualizer.  Used by chief supervisors.
        The directory will be created if it does not exist.
      summary_op: An `Operation` that returns a Summary for the event logs.
        Used by chief supervisors if a `logdir` was specified.  Defaults to the
        operation returned from summary.merge_all().  If `None`, summaries are
        not computed automatically.
      saver: A Saver object.  Used by chief supervisors if a `logdir` was
        specified.  Defaults to the saved returned by Saver().
        If `None`, the model is not saved automatically.
      global_step: An integer Tensor of size 1 that counts steps.  The value
        from 'global_step' is used in summaries and checkpoint filenames.
        Default to the op named 'global_step' in the graph if it exists, is of
        rank 1, size 1, and of type tf.int32 or tf.int64.  If `None` the global
        step is not recorded in summaries and checkpoint files.  Used by chief
        supervisors if a `logdir` was specified.
      save_summaries_secs: Number of seconds between the computation of
        summaries for the event log.  Defaults to 120 seconds.  Pass 0 to
        disable summaries.
      save_model_secs: Number of seconds between the creation of model
        checkpoints.  Defaults to 600 seconds.  Pass 0 to disable checkpoints.
      recovery_wait_secs: Number of seconds between checks that the model
        is ready.  Used by supervisors when waiting for a chief supervisor
        to initialize or restore the model.  Defaults to 30 seconds.
      stop_grace_secs: Grace period, in seconds, given to running threads to
        stop when `stop()` is called.  Defaults to 120 seconds.
      checkpoint_basename: The basename for checkpoint saving.
      session_manager: `SessionManager`, which manages Session creation and
        recovery. If it is `None`, a default `SessionManager` will be created
        with the set of arguments passed in for backwards compatibility.
      summary_writer: `SummaryWriter` to use or `USE_DEFAULT`.  Can be `None`
        to indicate that no summaries should be written.
      init_fn: Optional callable used to initialize the model. Called
        after the optional `init_op` is called.  The callable must accept one
        argument, the session being initialized.

    Returns:
      A `Supervisor`.
    """
    # Set default values of arguments.
    if graph is None:
      graph = ops.get_default_graph()
    with graph.as_default():
      self._init_ready_op(
          ready_op=ready_op, ready_for_local_init_op=ready_for_local_init_op)
      self._init_init_op(init_op=init_op, init_feed_dict=init_feed_dict)
      self._init_local_init_op(local_init_op=local_init_op)
      self._init_saver(saver=saver)
      self._init_summary_op(summary_op=summary_op)
      self._init_global_step(global_step=global_step)
    self._graph = graph
    self._meta_graph_def = meta_graph.create_meta_graph_def(
        graph_def=graph.as_graph_def(add_shapes=True),
        saver_def=self._saver.saver_def if self._saver else None)
    self._is_chief = is_chief
    self._coord = coordinator.Coordinator()
    self._recovery_wait_secs = recovery_wait_secs
    self._stop_grace_secs = stop_grace_secs
    self._init_fn = init_fn

    # Set all attributes related to checkpointing and writing events to None.
    # Afterwards, set them appropriately for chief supervisors, as these are
    # the only supervisors that can write checkpoints and events.
    self._logdir = None
    self._save_summaries_secs = None
    self._save_model_secs = None
    self._save_path = None
    self._summary_writer = None

    if self._is_chief:
      self._logdir = logdir
      self._save_summaries_secs = save_summaries_secs
      self._save_model_secs = save_model_secs
      if self._logdir:
        self._save_path = os.path.join(self._logdir, checkpoint_basename)
      if summary_writer is USE_DEFAULT:
        if self._logdir:
          self._summary_writer = _summary.FileWriter(self._logdir)
      else:
        self._summary_writer = summary_writer
      self._graph_added_to_summary = False

    self._init_session_manager(session_manager=session_manager)
    self._verify_setup()
    # The graph is not allowed to change anymore.
    # graph.finalize()

  def _init_session_manager(self, session_manager=None):
    if session_manager is None:
      self._session_manager = session_manager_mod.SessionManager(
          local_init_op=self._local_init_op,
          ready_op=self._ready_op,
          ready_for_local_init_op=self._ready_for_local_init_op,
          graph=self._graph,
          recovery_wait_secs=self._recovery_wait_secs)
    else:
      self._session_manager = session_manager

  def _get_first_op_from_collection(self, key):
    """Returns the first `Operation` from a collection.

    Args:
      key: A string collection key.

    Returns:
      The first Op found in a collection, or `None` if the collection is empty.
    """
    try:
      op_list = ops.get_collection(key)
      if len(op_list) > 1:
        tf_logging.info("Found %d %s operations. Returning the first one.",
                     len(op_list), key)
      if op_list:
        return op_list[0]
    except LookupError:
      pass

    return None



use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

def run(args, server):
    env = LabInterface(level='seekavoid_arena_01')

    if args.agent_id == "A3C":
        trainer = A3C(env, args.task)
    if args.agent_id == "A3CPC":
        trainer = A3CPC(env, args.task)
    if args.agent_id == "A3CFC":
        trainer = A3CFC(env, args.task)
    if args.agent_id == "A3CPCR":
        trainer = A3CPCR(env, args.task)
    if args.agent_id == "BasicQTDnet":
        trainer = ATDNet(env, args.task)
    if args.agent_id == "ConvTDNet":
        trainer = ConvTDAgent(env, args.task)
    else:
        print("Invalid agent specified!")

    # Variable names that start with "local" are not saved in checkpoints.
    if use_tf12_api:
        variables_to_save = [v for v in tf.global_variables() if v.name.startswith("learner")
                             ]
        init_op = tf.variables_initializer(variables_to_save)
        init_all_op = tf.global_variables_initializer()
    else:
        variables_to_save = [v for v in tf.all_variables() if v.name.startswith("learner")
                             ]
        init_op = tf.initialize_variables(variables_to_save)
        init_all_op = tf.initialize_all_variables()
    saver = FastSaver(variables_to_save, write_version=tf.train.SaverDef.V1)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

        print("here I am")

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    logdir = os.path.join(args.log_dir, 'train')

    if use_tf12_api:
        summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)
    else:
        summary_writer = tf.train.SummaryWriter(logdir + "_%d" % args.task)



    logger.info("Events directory: %s_%s", logdir, args.task)
    sv = NonFinalSupervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    num_global_steps = 100000000

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")

    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        saver2 = tf.train.Saver([v for v in tf.all_variables() if v.name.startswith("global")])
        trainer.beh_pi.restore(sess, saver2)
        sess.run(trainer.sync)

        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        logger.info("Starting training at step=%d", global_step)
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)
#            if global_step % 1000 == 0:
#                trainer.network.update_target_weights(sess)
#                print("updated target weights")

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)

def cluster_spec(num_workers, num_ps):
    """
More tensorflow setup for data parallelism
"""
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster

def main(_):
    """
Setting up Tensorflow for data parallel work
"""

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="ps", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/deepmindlabtest", help='Log directory path')
    parser.add_argument('--agent-id', default="A3C", help='Agent id')
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')

    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()
