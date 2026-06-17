import multiprocessing as mp
import os
import psutil
import subprocess
import time
import pkg_resources
import carla

class CarlaClient(carla.Client):  #对carla.Client 的轻量增强，核心目的是提升“连接就绪检测”和“加载地图”的可靠性
    LOAD_WORLD_ATTEMPTS = 5  #加载地图的最大重试次数，避免一次失败直接中断流程

    def test_connection(self):  #通过get_world()测试与CARLA服务器的连接是否就绪
        # Test the connection with the CARLA server by retrieving the world data.
        # This is a blocking call until the server responds or a RuntimeError occurs.
        self.get_world()

    def load_world(self, *args, **kwargs):  #在加载地图失败时进行有限次数的重试
        # Safely load a new map by retrying it a couple of times.
        world = None
        attempt = 0
        while world is None and attempt < self.LOAD_WORLD_ATTEMPTS:
            try:
                world = super().load_world(*args, **kwargs)
            except RuntimeError:
                world = None
                attempt += 1
        if world is None:
            raise RuntimeError("Could not load world")
        else:
            return world

 
class CarlaServer:
    CARLA_ROOT = os.environ.get('CARLA_ROOT') 
    carla_version_string = pkg_resources.get_distribution('carla').version
    x, y, z = carla_version_string.split('.')
    CARLA_VERSION = (int(x), int(y), int(z))

    def __init__(self, port=2000, offscreen=True, sound=False, launch_delay=30, launch_retries=3,
                 connect_timeout=10, connect_retries=3):
        self._port = port
        self._offscreen = offscreen  #控制是否离屏渲染
        self._sound = sound           # 是否开启音频输出
        self._launch_delay = launch_delay   #每次启动后等待的秒数
        self._launch_retries = launch_retries #服务器启动的最大重试次数；用于启动循环的上限
        self._connect_timeout = connect_timeout   #客户端连接的超时时间（秒）
        self._connect_retries = connect_retries  #客户端连接的最大重试次数；作为run_client的默认值
        self._server = None

    def launch(self, delay=None, retries=None):   # 实现服务器的“稳健启动与就绪检测”流程

        print(f'CARLA_VERSION: {self.CARLA_VERSION}')  #打印当前已安装的carla包版本，便于运行期诊断

        # Check if the server is already running
        if self.is_active:
            print("CARLA server is already running.")
            return

        # Set delays and retries
        if delay is None:   #用于每次启动后等待服务器初始化的时间
            delay = self._launch_delay
        if retries is None:   #retries 用于控制启动循环的最大尝试次数
            retries = self._launch_retries

        # Setup arguments and environment variables  准备启动参数和环境变量
        args = [f'-carla-port={self._port}']
        env = os.environ.copy()  #复制当前进程的环境变量到env，用于传给子进程（CARLA服务器），避免破坏当前进程环境
        if self._offscreen:  # 离屏渲染
            # Offscreen rendering (see https://carla.readthedocs.io/en/latest/adv_rendering_options/#off-screen-mode)
            if self.CARLA_VERSION >= (0, 9, 12):
                args.append('-RenderOffScreen')
            else:
                args.append('-opengl')
                env['DISPLAY'] = ''
        if not self._sound:  # 是否启用声音  不启用为False
            args.append('-nosound')
        carla_path = os.path.join(self.CARLA_ROOT, 'CarlaUE4.exe' if os.name == 'nt' else 'CarlaUE4.sh')
        cmd = [carla_path, *args]

        # Try launching the server  进入“尝试启动服务器”的逻辑块，后续会进行重试与就绪检测
        attempt = 0   # 初始化启动尝试计数
        self._server = None
        while self._server is None and attempt < retries:
            attempt += 1

            # Try to launch server and wait for delay seconds before attempting the first connection.
            print(f"Launching CARLA server (attempt {attempt}/{retries})")
            self._server = subprocess.Popen(cmd, env=env)  #使用准备好的命令与环境变量启动CARLA服务器进程，并保存进程句柄
            time.sleep(delay)     # 启动后等待预设秒数，给服务器初始化时间
            try:
                if self.is_active:
                    # Try to run _test_target in a client session with small timeout and many retries.
                    # This ensures the CARLA server is completely ready to handle further client connections.
                    self.run_client(self._test_target, worker_threads=1, timeout=5, retries=20, verbose=False)
            except RuntimeError:  #绪检测失败会抛出 RuntimeError
                # Server didn't respond to client connections, either it crashed or is unresponsive.
                # Kill server process if it didn't crash already and try again.
                self.kill()   
                self._server = None

            if not self.is_active:  #如果进程本身已退出（崩溃/异常），同样重置句柄并打印失败信息，继续下一次尝试
                # Server process terminated, retry launch
                self._server = None
                print("Launching CARLA server failed.")

        # If the server is still not active after all retries, give up.
        if not self.is_active:  #循环结束后若仍未处于运行状态，抛出“无法启动服务器”的异常，终止流程
            raise RuntimeError("Could not launch CARLA server.")
        else:
            print("CARLA server ready")

    def kill(self):  #不仅杀主进程，还递归杀掉子进程，避免模拟器残留
        if self.is_active:
            # CARLA server spawns child processes, make sure to kill them too (otherwise the simulator keeps running)
            children = psutil.Process(self._server.pid).children(recursive=True)
            for child in children:
                child.kill()
            self._server.kill()
            self._server = None

    @property
    def is_active(self):   #速判断“进程是否存活”
        return self._server is not None and self._server.poll() is None
    

    def run_client(self, target, worker_threads=0, timeout=None, retries=None, verbose=True):  #在“独立子进程”中执行客户端连接与业务逻辑
        # To prevent issues when clients are closed, it is recommended to launch the client in a separate process.
        # See https://github.com/carla-simulator/carla/issues/2789#issuecomment-689619998
        if timeout is None:  #分别使用实例上的默认连接超时与重试次数
            timeout = self._connect_timeout
        if retries is None:
            retries = self._connect_retries

        err_out, err_in = mp.Pipe(False)
        p = mp.Process(target=self._client_proc, args=(self._port, worker_threads, timeout, retries, verbose, target, err_in))
        try:
            p.start()
            # Receive exception or `False` from child process
            e = err_out.recv()
            if e:
                # Raise it again in main process
                raise e
        finally:
            p.join()  #无论成功或失败，最终都调用 p.join() 等待子进程退出

    @staticmethod    #子进程内的客户端连接与就绪验证逻辑：子进程中的“可靠连接 + 就绪检测 + 任务执行 + 异常回传”入口
    def _client_proc(port, worker_threads, timeout, retries, verbose=True, target=None, err=None):
        try:
            client = None
            attempt = 0
            while client is None and attempt < retries:
                attempt += 1
                if verbose:
                    print(f"Connecting to CARLA server (attempt {attempt}/{retries})")
                try:
                    client = CarlaClient('localhost', port, worker_threads)
                    client.set_timeout(timeout)
                    client.test_connection()  # Blocking call until server responds or RuntimeError occurs
                except RuntimeError:
                    # Client connection timed out, retry connection
                    client = None
                    if verbose:
                        print("Client connection timed out.")

            if client is None:
                exc = RuntimeError("Could not connect to CARLA server.")
                if err:
                    err.send(exc)
                    return
                else:
                    raise exc
            if verbose:
                print("Client ready")

            if target is None:  #业务执行
                return client
            else:
                try:
                    target(client)
                except Exception as exc:
                    # Send any exception through the error connection, such that it can be raised in the main process
                    if err:
                        err.send(exc)
                    # Raise it in child process as well to make debugging easier for users
                    raise exc
        finally:
            if err:
                # Always close error connection before returning
                err.send(False)
                err.close()

    @staticmethod   #用于作为客户端会话中的“就绪探测”目标函数
    def _test_target(client):  #是一个最小化的就绪探针：在客户端会话里阻塞获取世界，成功则判定服务器可用，失败则触发清理与重试，确保后续业务在稳定状态下进行
        # Test target to wait for the CARLA server to be ready for further client connections.
        client.test_connection()