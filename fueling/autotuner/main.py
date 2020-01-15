import asyncio
import json
import random
import requests
import threading
import time

from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, Colours
from tornado.web import RequestHandler
import tornado.httpserver
import tornado.ioloop


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, however, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its outputs values, as unknown.
    """
    time.sleep(random.randint(1, 7))
    return -(x ** 2) - (y - 1) ** 2 + 1


class BayesianOptimizationHandler(RequestHandler):
    """Basic functionality for NLP handlers."""
    _bo = BayesianOptimization(
        f=black_box_function,
        pbounds={"x": (-4, 4), "y": (-3, 3)}
    )
    _uf = UtilityFunction(kind="ucb", kappa=3, xi=1)

    def post(self):
        """Deal with incoming requests."""
        body = tornado.escape.json_decode(self.request.body)

        try:
            self._bo.register(
                params=body["params"],
                target=body["target"],
            )
            print(f"BO has registered: {len(self._bo.space)} points.", end="\n\n")
        except KeyError:
            pass
        finally:
            suggested_params = self._bo.suggest(self._uf)

        self.write(json.dumps(suggested_params))


def run_optimization_app():
    asyncio.set_event_loop(asyncio.new_event_loop())
    handlers = [
        (r"/bayesian_optimization", BayesianOptimizationHandler),
    ]
    server = tornado.httpserver.HTTPServer(
        tornado.web.Application(handlers)
    )
    server.listen(9009)
    tornado.ioloop.IOLoop.instance().start()


def run_optimizer():
    global optimizers_config
    config = optimizers_config.pop()
    name = config["name"]
    colour = config["colour"]

    register_data = {}
    max_target = None
    for _ in range(10):
        status = name + f" wants to register: {register_data}.\n"

        resp = requests.post(
            url="http://localhost:9009/bayesian_optimization",
            json=register_data,
        ).json()
        target = black_box_function(**resp)

        register_data = {
            "params": resp,
            "target": target,
        }

        if max_target is None or target > max_target:
            max_target = target

        status += name + f" got {target} as target.\n"
        status += name + f" will to register next: {register_data}.\n"
        print(colour(status), end="\n")

    global results
    results.append((name, max_target))
    print(colour(name + " is done!"), end="\n\n")


if __name__ == "__main__":
    ioloop = tornado.ioloop.IOLoop.instance()
    optimizers_config = [
        {"name": "optimizer 1", "colour": Colours.red},
        {"name": "optimizer 2", "colour": Colours.green},
        {"name": "optimizer 3", "colour": Colours.blue},
    ]

    app_thread = threading.Thread(target=run_optimization_app)
    app_thread.daemon = True
    app_thread.start()

    targets = (
        run_optimizer,
        run_optimizer,
        run_optimizer
    )
    optimizer_threads = []
    for target in targets:
        optimizer_threads.append(threading.Thread(target=target, daemon=True))
        optimizer_threads[-1].start()

    results = []
    for optimizer_thread in optimizer_threads:
        optimizer_thread.join()

    for result in results:
        print(result[0], f"found a maximum value of: {result[1]}")

    ioloop.stop()
