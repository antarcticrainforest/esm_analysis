import tqdm
from tornado.ioloop import IOLoop
from distributed.utils import LoopRunner, is_kernel
from distributed.client import futures_of
from distributed.diagnostics.progressbar import ProgressBar

class ProgressBar(ProgressBar):
    def __init__(
        self,
        keys,
        scheduler=None,
        interval="100ms",
        loop=None,
        complete=True,
        start=True,
        **tqdm_kwargs
    ):
        super(ProgressBar, self).__init__(
            keys, scheduler, interval, complete)
        self.tqdm = tqdm.tqdm(keys, **tqdm_kwargs)
        self.loop = loop or IOLoop()

        if start:
            loop_runner = LoopRunner(self.loop)
            loop_runner.run_sync(self.listen)

    def _draw_bar(self, remaining, all, **kwargs):
        update_ct = (all - remaining) - self.tqdm.n
        self.tqdm.update(update_ct)

    def _draw_stop(self, **kwargs):
        self.tqdm.close()

class NotebookProgress(ProgressBar):
    def __init__(
        self,
        keys,
        scheduler=None,
        interval="100ms",
        loop=None,
        complete=True,
        start=True,
        **tqdm_kwargs
    ):
        super(NotebookProgress, self).__init__(
            keys, scheduler, interval, complete)
        self.tqdm = tqdm.tqdm_notebook(keys, **tqdm_kwargs)
        self.loop = loop or IOLoop()

        if start:
            loop_runner = LoopRunner(self.loop)
            loop_runner.run_sync(self.listen)

    def _draw_bar(self, remaining, all, **kwargs):
        update_ct = (all - remaining) - self.tqdm.n
        self.tqdm.update(update_ct)

    def _draw_stop(self, **kwargs):
        self.tqdm.close()
