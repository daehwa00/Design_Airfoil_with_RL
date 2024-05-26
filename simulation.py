import os
import subprocess


def run_simulation(verbose=False):
    """
    OpenFOAM을 사용하여 시뮬레이션을 실행합니다.
    """
    original_directory = setup_simulation_environment()
    clean_simulation(verbose)
    move_block_mesh_dict_and_control_dict(verbose)
    generate_mesh(verbose)
    remove_processor_directories(verbose)
    decompose_mesh(verbose)
    set_permissions(verbose)
    run_parallel_simulation(verbose)
    Cd, Cl = read_force_data()
    remove_forceCoeffs(verbose)
    os.chdir(original_directory)  # 원래 디렉토리로 돌아갑니다.
    return Cd, Cl


def setup_simulation_environment():
    """
    시뮬레이션 디렉토리로 작업 디렉토리를 변경합니다.
    """
    simulation_directory = "~/OpenFOAM/daehwa-11/run/airfoil"
    original_directory = os.getcwd()  # 현재 작업 디렉토리를 저장합니다.
    os.chdir(os.path.expanduser(simulation_directory))
    return original_directory  # 원래 디렉토리를 반환합니다.


def clean_simulation(verbose):
    """
    시뮬레이션 디렉토리를 정리합니다.
    """
    run_command("sh ./Allclean", verbose)


def move_block_mesh_dict_and_control_dict(verbose):
    """
    blockMeshDict 파일을 적절한 위치로 이동합니다.
    """
    source_path = "~/Documents/3D-propeller-Design/blockMeshDict"
    destination_directory = "~/OpenFOAM/daehwa-11/run/airfoil/system"
    run_command(
        f"mv {os.path.expanduser(source_path)} {os.path.expanduser(destination_directory)}",
        verbose,
    )

    source_path = "~/Documents/3D-propeller-Design/controlDict"
    run_command(
        f"mv {os.path.expanduser(source_path)} {os.path.expanduser(destination_directory)}",
        verbose,
    )

    source_path = "~/Documents/3D-propeller-Design/U"
    destination_directory = "~/OpenFOAM/daehwa-11/run/airfoil/0"
    run_command(
        f"mv {os.path.expanduser(source_path)} {os.path.expanduser(destination_directory)}",
        verbose,
    )


def generate_mesh(verbose):
    """
    blockMesh를 사용하여 메시를 생성합니다.
    """
    run_command("blockMesh", verbose)


def set_permissions(verbose):
    """
    points 파일에 대한 권한을 설정합니다.
    """
    points_path = "~/OpenFOAM/daehwa-11/run/airfoil/constant/polyMesh/points"
    run_command(f"chmod 777 {os.path.expanduser(points_path)}", verbose)


def remove_processor_directories(verbose):
    """
    기존의 processor 디렉토리를 삭제합니다.
    """
    run_command("rm -rf processor*", verbose)


def decompose_mesh(verbose):
    """
    메시를 여러 부분으로 나누어 병렬 처리를 준비합니다.
    """
    run_command("decomposePar", verbose)


def run_parallel_simulation(verbose):
    """
    병렬로 시뮬레이션을 실행합니다.
    """
    run_command(
        "mpirun --oversubscribe -np 20 foamRun -solver incompressibleFluid -parallel",
        verbose,
    )
    run_command("reconstructPar", verbose)
    run_command("rm -rf processor*", verbose)


def read_force_data():
    """
    forceCoeffs.dat 파일을 읽어 마지막 줄의 시간, 항력 계수, 양력 계수를 반환합니다.
    """
    result_file_path = (
        "~/OpenFOAM/daehwa-11/run/airfoil/postProcessing/forceCoeffs/0/forceCoeffs.dat"
    )
    with open(os.path.expanduser(result_file_path), "r") as file:
        lines = file.readlines()

    last_line = lines[-1].split()
    Cd, Cl = float(last_line[2]), float(last_line[3])
    return Cd, Cl


def remove_forceCoeffs(verbose):
    """
    forceCoeffs 디렉토리를 삭제합니다.
    """
    run_command("rm -rf postProcessing/forceCoeffs/0/forceCoeffs.dat", verbose)


def run_command(command, verbose):
    """
    명령어를 실행하고, verbose가 True일 경우 명령어의 출력을 표시합니다.
    """
    process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if verbose:
        print(process.stdout.decode())
        print(process.stderr.decode())


# 예제 사용법:
# Cd, Cl = run_simulation(verbose=True)
