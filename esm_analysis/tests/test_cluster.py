"""Test the MPICluster module."""
from pathlib import Path

import pytest

from esm_analysis import MPICluster


def test_createCluster(mock_slurm, mock_workdir, monkeypatch):
    """Test creating a new cluster."""
    import os
    Cluster = MPICluster.slurm('mh0731', 'gpu', workdir=mock_workdir)
    assert Cluster.job_id == os.environ['JOB_ID']
    assert str(Cluster.workdir) == mock_workdir
    status = 'Q'
    assert Cluster.status == 'Q'
    monkeypatch.setenv('STATUS', '')
    assert Cluster.status is None
    assert Cluster.__repr__() == 'No cluster running'


def test_loadCluster(mock_slurm, mock_workdir, monkeypatch):
    """Test loading an existing cluster."""
    import os
    monkeypatch.setenv('STATUS', 'R')
    Cluster = MPICluster.load(mock_workdir)
    assert Cluster.status == 'R'
    assert Cluster._batch_system.type == 'slurm'
    assert Cluster.job_id == os.environ['JOB_ID']
    assert Cluster.__repr__() == 'Running: time: 0:00 nodes: 2'
    html_txt = '''<p> <span style="color:MediumSeaGreen;">Running</span>:
                  time: 0:00
                  nodes: 2</p>'''
    assert Cluster._repr_html_() == html_txt
    Cluster.close()
    assert Cluster.job_id is None
    Cluster.close()
    assert Cluster.status is None
    assert Cluster._repr_html_() == '<p>No cluster running<p>'


def test_loadNonExistingCluster(mock_slurm):
    """Test loading a cluster without information."""
    with pytest.raises(ValueError):
        MPICluster.load('.')
