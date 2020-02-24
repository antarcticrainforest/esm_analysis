from datetime import datetime
import json
from pathlib import Path
from subprocess import run, PIPE
from tempfile import TemporaryDirectory


__all__ = ['MPICluster']

slurm_directive="""#!/bin/bash
#=============================================================================
# =====================================
# mistral batch job parameters
#-----------------------------------------------------------------------------
#SBATCH --account={account}
#SBATCH --job-name={name}
#SBATCH --partition={queue}
#SBATCH -D {workdir}
#SBATCH --output={workdir}/LOG_dask.%j.o
#SBATCH --error={workdir}/LOG_dask.%j.o
#SBATCH --exclusive
#SBATCH --time={walltime}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem=140G
#SBATCH -n {nworkers}

"""
_script="""
rm -r worker-*
rm scheduler.json
rm global.lock
rm purge.lock
rm LOG*.o

ulimit -c 0

{job_extra}

# Settings for OpenMPI and MXM (MellanoX Messaging)
# library
export OMPI_MCA_pml=cm
export OMPI_MCA_mtl=mxm
export OMPI_MCA_mtl_mxm_np=0
export MXM_RDMA_PORTS=mlx5_0:1
export MXM_LOG_LEVEL=ERROR
# Disable GHC algorithm for collective communication
export OMPI_MCA_coll=^ghc

{run_cmd} dask-mpi --no-nanny --scheduler-file scheduler.json

"""

class BatchBase:
    type = None
    submit_cmd = 'qsub'
    cancel_cmd = 'qdel'
    check_cmd = 'qstat'
    run_cmd = 'mpirun'


class Slurm(BatchBase):
    type = 'slurm'
    submit_cmd = 'sbatch'
    cancel_cmd = 'scancel'
    check_cmd =  'squeue'
    run_cmd = 'srun -l --cpu_bind=threads --distribution=block:cyclic --propagate=STAC'

    def cancel(self, job_id):
        """Close down a cluster with a given job_id."""

        if job_id is None:
            return
        run([self.cancel_cmd, job_id], stdout=PIPE, check=True)

    def check(self, job_id, simple=True, color=True):
        """Check the status of a running cluster."""

        if job_id is None:
            return None, None, None
        res = run([self.check_cmd, '-j {}'.format(job_id)],
                   stdout=PIPE).stdout.decode('utf-8').split('\n')
        if  len(res) < 2:
            return None, None, None
        status = [line.split() for line in res]
        table = dict(zip(status[0], status[1][:len(status[0])]))

        status_l = dict(PD='Queueing', R='Running', F='Failed')
        return status_l[table['ST']], table['TIME'], table['NODES']

class MPICluster:
    """Create an instance of a Worker Cluster using."""


    def close(self):
        """Close down the running cluster."""

        self.batch_system.cancel(self.job_id)
        self.job_id = None
        self._write_json()

    @property
    def status(self):
        """Check the status of the running cluster."""

        status, _, _ = self.batch_system.check(self.job_id)

        try:
            return status[0].upper()
        except TypeError:
            return None


    def __repr__(self):

        status, time, nodes = self.batch_system.check(self.job_id)
        if status is None:
            return 'No cluster running'

        return '{}: time: {} nodes: {}'.format(status, time, nodes)

    def _repr_html_(self):

        status, time, nodes = self.batch_system.check(self.job_id)
        colors = dict(Queueing='DodgerBlue',
                      Fail='Tomato',
                      Running='MediumSeaGreen')
        if status is None:
            return '<p>No cluster running<p>'
        color = colors[status]

        return '''<p> <span style="color:{color};">{status}</span>:
                  time: {time}
                  nodes: {nodes}</p>'''.format(color=color,
                                               status=status,
                                               time=time,
                                               nodes=nodes)

    @property
    def script_path(self):
        """Return the path of the script that is/was submitted."""

        return  Path(self.workdir) / 'scheduler.sh'

    def _write_script(self):

        with open(str(self.script_path), 'w') as f:
                f.write(self.job_script)
        self.script_path.chmod(0o755)

    def _write_json(self):

        _json_data= dict(job_id=self.job_id,
                           workdir=str(self.workdir),
                           job_script=self.job_script,
                           batch_system=self.batch_system.type,
                           datetime=self.submit_time.isoformat())
        with (self.workdir / 'cluster.json').open('w') as f:
            json.dump(_json_data, f, indent=3, sort_keys=True)

    @staticmethod
    def _load(workdir):
        try:
            with (workdir / 'cluster.json').open('r') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            raise ValueError('Cluster has not been created.')

        json_data['datetime'] =  datetime.strptime(json_data['datetime'],
                                                   '%Y-%m-%dT%H:%M:%S.%f')
        json_data['workdir'] = Path(json_data['workdir'])
        return json_data

    @classmethod
    def load(cls, workdir):
        """Load the information of a running cluster."""

        workdir = Path(workdir)
        _json_data = cls._load(workdir)
        lookup = dict(slurm=Slurm)
        batch_system = lookup[_json_data['batch_system']]()
        script = _json_data['job_script']
        job_id = _json_data['job_id']
        if job_id is None:
            raise ValueError('Cluster was closed, submit a new one')
        submit_time = _json_data['datetime']

        return cls(script, workdir, submit_time=submit_time,
                   job_id=job_id, batch_system=batch_system)

    def _submit(self):

        res = run([self.batch_system.submit_cmd, str(self.script_path)],
                  cwd=str(self.workdir), stdout=PIPE, check=True)
        job_id, _, _cluster = res.stdout.decode('utf-8').strip().partition(';')
        return job_id.split(" ")[-1]

    def __init__(self, script, workdir, submit_time=None, batch_system=None,
                 job_id=None):

        """Create a cluster using a given submit script."""

        self.job_script = script
        self.submit_time = submit_time
        self.job_id = job_id
        self.batch_system = batch_system
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        if self.submit_time is None:
            self._write_script()
            self.job_id = self._submit()
            self.submit_time = datetime.now()
            self._write_json()

    @classmethod
    def slurm(cls, account, queue, *,
                 workdir=None,
                 walltime='01:00:00',
                 cpus_per_task=48,
                 name='dask_job',
                 nworkers=1,
                 job_extra=None):
        """Create an MPI cluster using slurm.

        Parameters
        ----------

        account: str
            Account name

        queue: str
            partition job should be submitted to

        walltime: str, optional (default: '01:00:00')
            lenth of the job

        name: str, optional (default: dask_job)
            name of the job

        workdir: str, optional (default: None)
            name of the workdirectory, if None is given, a temporary directory is
            used.

        cpus_per_task: int, optional (default: 48)
            number of cpus per node

        nworkers: int, optional (default: 1)
            number of nodes used in the job

        job_extra: str, optional (default: None)
            additional commands that should be executed in the run sript

        Return
        ------
        str: job_id of the submitted job
        """

        job_extra = job_extra or ''
        workdir = workdir or TemporaryDirectory().name
        workdir = Path(workdir)
        batch_system = Slurm()
        script = slurm_directive.format(
                account=account,
                workdir=workdir,
                name=name,
                cpus_per_task=cpus_per_task,
                nworkers=nworkers+1,
                walltime=walltime,
                queue=queue) + _script.format(run_cmd=batch_system.run_cmd,
                                              job_extra=job_extra)

        return cls(script, workdir, batch_system=batch_system)
