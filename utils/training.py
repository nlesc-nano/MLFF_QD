import os
import logging
import numpy as np
import torch
from ase import Atoms
import schnetpack as spk
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.data_processing import read_homo_lumo_and_gaps_to_dataframe, read_eigenvalues_to_dataframe
from utils.model import setup_model
from utils.helpers import load_config, get_optimizer_class, get_scheduler_class
import schnetpack.transform as trn

# Configure logging
logging.basicConfig(level=logging.INFO)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(config):
    try:
        data = np.load(config['settings']['data']['dataset_path'])
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {config['settings']['data']['dataset_path']}")
        raise

    output_type = config['settings']['data']['output_type']
    include_homo_lumo = output_type == 2
    include_bandgap = output_type == 3
    include_eigenvalues_vector = output_type == 4

    df_eigenvalues = None
    if include_homo_lumo or include_bandgap or include_eigenvalues_vector:
        data_hdf = config['settings']['data']['hdf5_file']

    if include_homo_lumo or include_bandgap:
        df_eigenvalues = read_homo_lumo_and_gaps_to_dataframe(data_hdf)
        
    if include_eigenvalues_vector:
        eigenvalue_labels = config['settings']['data']['eigenvalue_labels']
        df_eigenvalues = read_eigenvalues_to_dataframe(data_hdf, eigenvalue_labels)

    return data, df_eigenvalues, include_homo_lumo, include_bandgap, include_eigenvalues_vector

def preprocess_data(data, df_eigenvalues, include_homo_lumo, include_bandgap, include_eigenvalues_vector):
    numbers = data["z"]
    atoms_list = []
    property_list = []
    
    positions_array = np.array(data["R"])
    energies_array = np.array(data["E"])
    forces_array = np.array(data["F"])
    
    if include_homo_lumo or include_bandgap or include_eigenvalues_vector:
        df_eigenvalues.set_index('index', inplace=True, drop=False)
        homo_values = df_eigenvalues["HOMO_Energy (eV)"].values if include_homo_lumo else None
        lumo_values = df_eigenvalues["LUMO_Energy (eV)"].values if include_homo_lumo else None
        bandgap_values = df_eigenvalues["HOMO_LUMO_Gap (eV)"].values if include_bandgap else None

    if include_eigenvalues_vector:
        eigenvalue_columns = [col for col in df_eigenvalues.columns if col not in ['index', 'HOMO_Energy (eV)', 'LUMO_Energy (eV)', 'HOMO_LUMO_Gap (eV)']]
        eigenvalues_vectors = df_eigenvalues[eigenvalue_columns].values.astype(np.float32)

    for idx in range(len(positions_array)):
        ats = Atoms(positions=positions_array[idx], numbers=numbers)
        properties = {
            'energy': np.array([energies_array[idx]], dtype=np.float32),
            'forces': forces_array[idx].astype(np.float32)
        }

        if include_homo_lumo:
            properties.update({
                'homo': np.array([homo_values[idx]], dtype=np.float32),
                'lumo': np.array([lumo_values[idx]], dtype=np.float32)
            })

        if include_bandgap:
            properties['bandgap'] = np.array([bandgap_values[idx]], dtype=np.float32)

        if include_eigenvalues_vector:
            properties['eigenvalues_vector'] = eigenvalues_vectors[idx].reshape(1, -1)

        atoms_list.append(ats)
        property_list.append(properties)

    return atoms_list, property_list

def setup_logging_and_dataset(config, atoms_list, property_list, include_homo_lumo, include_bandgap, include_eigenvalues_vector):
    folder = config['settings']['logging']['folder']
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "lightning_logs"), exist_ok=True)

    db_path = os.path.join(folder, config['settings']['general']['database_name'])
    if os.path.exists(db_path):
        os.remove(db_path)

    property_units = {
        'energy': config['settings']['model']['property_unit_dict']['energy'],
        'forces': config['settings']['model']['property_unit_dict']['forces']
    }
    if include_homo_lumo:
        property_units.update({'homo': 'eV', 'lumo': 'eV'})
    if include_bandgap:
        property_units.update({'bandgap': 'eV'})
    if include_eigenvalues_vector:
        property_units.update({'eigenvalues_vector': 'eV'})

    new_dataset = spk.data.ASEAtomsData.create(
        db_path,
        distance_unit=config['settings']['model']['distance_unit'],
        property_unit_dict=property_units
    )

    new_dataset.add_systems(property_list, atoms_list)
    logging.info(f"Dataset created at {db_path}")
    
    return new_dataset, property_units 

def prepare_transformations(config, include_homo_lumo, include_bandgap):
    transformations = [
        trn.ASENeighborList(cutoff=config['settings']['model']['cutoff']),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ]
    if include_homo_lumo:
        transformations.append(trn.RemoveOffsets("homo", remove_mean=True, remove_atomrefs=False))
        transformations.append(trn.RemoveOffsets("lumo", remove_mean=True, remove_atomrefs=False))
    if include_bandgap:
        transformations.append(trn.RemoveOffsets("bandgap", remove_mean=True, remove_atomrefs=False))

    return transformations

def setup_data_module(config, db_path, transformations, property_units):
    custom_data = spk.data.AtomsDataModule(
        db_path,
        batch_size=config['settings']['training']['batch_size'],
        distance_unit=config['settings']['model']['distance_unit'],
        property_units=property_units,  # Use property_units directly
        num_train=config['settings']['training']['num_train'],
        num_val=config['settings']['training']['num_val'],
        transforms=transformations,
        num_workers=config['settings']['training']['num_workers'],
        pin_memory=config['settings']['training']['pin_memory']
    )

    custom_data.prepare_data()
    custom_data.setup()
    logging.info("Data module prepared and set up")
    
    return custom_data

def setup_task_and_trainer(config, nnpot, outputs, folder):
    optimizer_name = config['settings']['training']['optimizer']['type']
    scheduler_name = config['settings']['training']['scheduler']['type']

    optimizer_cls = get_optimizer_class(optimizer_name)
    scheduler_cls = get_scheduler_class(scheduler_name)
    
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=outputs,
        optimizer_cls=optimizer_cls,
        optimizer_args={"lr": config['settings']['training']['optimizer']['lr']},
        scheduler_cls=scheduler_cls,
        scheduler_args={"mode": "min", "factor": config['settings']['training']['scheduler']['factor'],
                        "patience": config['settings']['training']['scheduler']['patience'],
                        "verbose": config['settings']['training']['scheduler']['verbose']},
        scheduler_monitor=config['settings']['logging']['monitor']
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=folder)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(folder, config['settings']['logging']['checkpoint_dir']),
            save_top_k=1,
            monitor=config['settings']['logging']['monitor']
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=folder,
        max_epochs=config['settings']['training']['max_epochs'],
        accelerator=config['settings']['training']['accelerator'],
        precision=config['settings']['training']['precision'],
        devices=config['settings']['training']['devices']
    )
    
    logging.info("Task and trainer set up")
    return task, trainer
       
def main(args):
    config = load_config(args.config)
    set_seed(config['settings']['general']['seed'])

    data, df_eigenvalues, include_homo_lumo, include_bandgap, include_eigenvalues_vector = load_data(config)
    atoms_list, property_list = preprocess_data(data, df_eigenvalues, include_homo_lumo, include_bandgap, include_eigenvalues_vector)
    new_dataset, property_units = setup_logging_and_dataset(config, atoms_list, property_list, include_homo_lumo, include_bandgap, include_eigenvalues_vector)
    
    transformations = prepare_transformations(config, include_homo_lumo, include_bandgap)
    custom_data = setup_data_module(config, os.path.join(config['settings']['logging']['folder'], config['settings']['general']['database_name']), transformations, property_units)
    
    nnpot, outputs = setup_model(config, include_homo_lumo, include_bandgap, include_eigenvalues_vector)
    task, trainer = setup_task_and_trainer(config, nnpot, outputs, config['settings']['logging']['folder'])
    
    logging.info("Starting training")
    trainer.fit(task, datamodule=custom_data)