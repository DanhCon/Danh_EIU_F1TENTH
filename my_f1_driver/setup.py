from setuptools import find_packages, setup

package_name = 'my_f1_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='danh',
    maintainer_email='danh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_drive = my_f1_driver.drive_node:main',
            'pp_basic = my_f1_driver.pp_basic:main',
            'rrt_remake = my_f1_driver.RRT_remake:main',
            'claude = my_f1_driver.pp_claude:main',
            'claude_1400 = my_f1_driver.RRT_remake_claude:main',
            'ne_vat_can_new_tet = my_f1_driver.pp_ne_vat_can_25_2:main',
            'xe_2 = my_f1_driver.xe_2:main',
            'mpc = my_f1_driver.mpc_basic:main',
            'mpc_kinematic = my_f1_driver.mpc_kinematic:main',
            'mpc_2 = my_f1_driver.mpc_2:main',
            'mpc_lidar = my_f1_driver.mpc_lidar:main',
            'FTG = my_f1_driver.FTG:main',
            'dispart = my_f1_driver.paper.dispart:main',
            'mppi_basic = my_f1_driver.MPPI.mppi_basic:main',
            'mppi_real = my_f1_driver.MPPI.mppi_real:main',
            'mpc_from_real_900 = my_f1_driver.MPC.mpc_chuyen_doi_from_real:main',
            'PID_basic = my_f1_driver.PID_basic:main',
            'mppi_nhat = my_f1_driver.MPPI.mppi_nhat:main',
            'mpc_adaptive = my_f1_driver.MPC.mpc_adaptive:main'
            
        ],
    },
)
