_base_ = './xview_rtmdet_m_smoke20_baseline_gpu89.py'

custom_imports = dict(
    imports=['M_AD.models.dense_heads.gs3c_rtmdet_head'],
    allow_failed_imports=False)

work_dir = 'work_dirs/gs3c_xview_network_smoke_20260619/gs3c_g1_light_eval'

model = dict(
    bbox_head=dict(
        type='GSRRotatedRTMDetSepBNHead',
        gaussian_semantic_scale=dict(
            enable=True,
            class_area_priors_csv='work_dirs/gs3c_dataset_inventory_20260619/xview_new_800_600_train_labeled_area_priors.csv',
            class_names=[
                'Aircraft_Hangar', 'Barge', 'Building', 'Bus',
                'Cargo_Truck', 'Cargo_or_Container_Car', 'Cement_Mixer',
                'Construction_Site', 'Container_Crane', 'Container_Ship',
                'Crane_Truck', 'Damaged_Building', 'Dump_Truck',
                'Engineering_Vehicle', 'Excavator', 'Facility', 'Ferry',
                'Fishing_Vessel', 'Fixed-wing_Aircraft', 'Flat_Car',
                'Front_loader_or_Bulldozer', 'Ground_Grader', 'Haul_Truck',
                'Helicopter', 'Helipad', 'Hut_or_Tent', 'Locomotive',
                'Maritime_Vessel', 'Mobile_Crane', 'Motorboat',
                'Oil_Tanker', 'Passenger_Car', 'Passenger_Vehicle',
                'Passenger_or_Cargo_Plane', 'Pickup_Truck', 'Pylon',
                'Railway_Vehicle', 'Reach_Stacker', 'Sailboat',
                'Scraper_or_Tractor', 'Shed', 'Shipping_Container',
                'Shipping_container_lot', 'Small_Aircraft', 'Small_Car',
                'Storage_Tank', 'Straddle_Carrier', 'Tank_car', 'Tower',
                'Tower_crane', 'Trailer', 'Truck', 'Truck_Tractor',
                'Truck_Tractor_with_Box_Trailer',
                'Truck_Tractor_with_Flatbed_Trailer',
                'Truck_Tractor_with_Liquid_Tank', 'Tugboat',
                'Utility_Truck', 'Vehicle_Lot', 'Yacht'
            ],
            mode='continuous_logit_energy',
            z0=5.0,
            beta=0.34657359028,
            adapter_hidden=64,
            adapter_nonpositive_delta=True,
            preserve_s3c_guard=True,
            guard_lambda=0.25,
            domain_mode='closed_set')))
