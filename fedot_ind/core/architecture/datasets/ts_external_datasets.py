# from fedot_ind.core.architecture.settings.computational import backend_methods as np
# try: from urllib import urlretrieveda
# except ImportError: from urllib.request import urlretrieve
# import shutil
# from distutils.util import strtobool
#
# def get_predefined_splits(*xs):
#     '''xs is a list with X_train, X_valid, ...'''
#     splits_ = []
#     start = 0
#     for x in xs:
#         splits_.append(L(list(np.arange(start, start + len(x)))))
#         start += len(x)
#     return tuple(splits_)
#
# def get_UCR_univariate_list():
#     return [
#         'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
#         'AllGestureWiimoteZ', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken',
#         'BME', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration',
#         'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY',
#         'CricketZ', 'Crop', 'DiatomSizeReduction',
#         'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
#         'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame',
#         'DodgerLoopWeekend', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays',
#         'ElectricDevices', 'EOGHorizontalSignal', 'EOGVerticalSignal',
#         'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
#         'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain',
#         'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
#         'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan',
#         'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham',
#         'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
#         'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
#         'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2',
#         'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
#         'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
#         'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
#         'MoteStrain', 'NonInvasiveFetalECGThorax1',
#         'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf',
#         'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ',
#         'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'Plane',
#         'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
#         'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
#         'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
#         'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
#         'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
#         'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
#         'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl',
#         'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
#         'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
#         'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine',
#         'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
#     ]
#
#
# def get_UCR_multivariate_list():
#     return [
#         'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
#         'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms',
#         'Epilepsy', 'ERing', 'EthanolConcentration', 'FaceDetection',
#         'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
#         'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
#         'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports',
#         'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits',
#         'StandWalkJump', 'UWaveGestureLibrary'
#     ]
#
# def get_Monash_regression_list():
#     return sorted([
#         "AustraliaRainfall", "HouseholdPowerConsumption1",
#         "HouseholdPowerConsumption2", "BeijingPM25Quality",
#         "BeijingPM10Quality", "Covid3Month", "LiveFuelMoistureContent",
#         "FloodModeling1", "FloodModeling2", "FloodModeling3",
#         "AppliancesEnergy", "BenzeneConcentration", "NewsHeadlineSentiment",
#         "NewsTitleSentiment", "IEEEPPG",
#         #"BIDMC32RR", "BIDMC32HR", "BIDMC32SpO2", "PPGDalia" # Cannot be downloaded
#     ])
#
# def get_UCR_data(dsid, path='.', parent_dir='data/UCR', on_disk=True, mode='c', Xdtype='float32', ydtype=None,
#                  return_split=True, split_data=True,
#                  force_download=False, verbose=False):
#     dsid_list = [ds for ds in UCR_list if ds.lower() == dsid.lower()]
#     assert len(dsid_list) > 0, f'{dsid} is not a UCR dataset'
#     dsid = dsid_list[0]
#     return_split = return_split and split_data  # keep return_split for compatibility. It will be replaced by split_data
#     if dsid in ['InsectWingbeat']:
#         warnings.warn(f'Be aware that download of the {dsid} dataset is very slow!')
#     pv(f'Dataset: {dsid}', verbose)
#     full_parent_dir = Path(path) / parent_dir
#     full_tgt_dir = full_parent_dir / dsid
#     #     if not os.path.exists(full_tgt_dir): os.makedirs(full_tgt_dir)
#     full_tgt_dir.parent.mkdir(parents=True, exist_ok=True)
#     if force_download or not all([os.path.isfile(f'{full_tgt_dir}/{fn}.npy') for fn in
#                                   ['X_train', 'X_valid', 'y_train', 'y_valid', 'X', 'y']]):
#         # Option A
#         src_website = 'http://www.timeseriesclassification.com/aeon-toolkit'
#         decompress_from_url(f'{src_website}/{dsid}.zip', target_dir=full_tgt_dir, verbose=verbose)
#         if dsid == 'DuckDuckGeese':
#             with zipfile.ZipFile(Path(f'{full_parent_dir}/DuckDuckGeese/DuckDuckGeese_ts.zip'), 'r') as zip_ref:
#                 zip_ref.extractall(Path(parent_dir))
#         if not os.path.exists(full_tgt_dir / f'{dsid}_TRAIN.ts') or not os.path.exists(
#                 full_tgt_dir / f'{dsid}_TRAIN.ts') or \
#                 Path(full_tgt_dir / f'{dsid}_TRAIN.ts').stat().st_size == 0 or Path(
#             full_tgt_dir / f'{dsid}_TEST.ts').stat().st_size == 0:
#             print('It has not been possible to download the required files')
#             if return_split:
#                 return None, None, None, None
#             else:
#                 return None, None, None
#
#         pv('loading ts files to dataframe...', verbose)
#         X_train_df, y_train = _ts2df(full_tgt_dir / f'{dsid}_TRAIN.ts')
#         X_valid_df, y_valid = _ts2df(full_tgt_dir / f'{dsid}_TEST.ts')
#         pv('...ts files loaded', verbose)
#         pv('preparing numpy arrays...', verbose)
#         X_train_ = []
#         X_valid_ = []
#         for i in progress_bar(range(X_train_df.shape[-1]), display=verbose, leave=False):
#             X_train_.append(stack_pad(X_train_df[f'dim_{i}']))  # stack arrays even if they have different lengths
#             X_valid_.append(stack_pad(X_valid_df[f'dim_{i}']))  # stack arrays even if they have different lengths
#         X_train = np.transpose(np.stack(X_train_, axis=-1), (0, 2, 1))
#         X_valid = np.transpose(np.stack(X_valid_, axis=-1), (0, 2, 1))
#         X_train, X_valid = match_seq_len(X_train, X_valid)
#
#         np.save(f'{full_tgt_dir}/X_train.npy', X_train)
#         np.save(f'{full_tgt_dir}/y_train.npy', y_train)
#         np.save(f'{full_tgt_dir}/X_valid.npy', X_valid)
#         np.save(f'{full_tgt_dir}/y_valid.npy', y_valid)
#         np.save(f'{full_tgt_dir}/X.npy', concat(X_train, X_valid))
#         np.save(f'{full_tgt_dir}/y.npy', concat(y_train, y_valid))
#         del X_train, X_valid, y_train, y_valid
#         delete_all_in_dir(full_tgt_dir, exception='.npy')
#         pv('...numpy arrays correctly saved', verbose)
#
#     mmap_mode = mode if on_disk else None
#     X_train = np.load(f'{full_tgt_dir}/X_train.npy', mmap_mode=mmap_mode)
#     y_train = np.load(f'{full_tgt_dir}/y_train.npy', mmap_mode=mmap_mode)
#     X_valid = np.load(f'{full_tgt_dir}/X_valid.npy', mmap_mode=mmap_mode)
#     y_valid = np.load(f'{full_tgt_dir}/y_valid.npy', mmap_mode=mmap_mode)
#
#     if return_split:
#         if Xdtype is not None:
#             X_train = X_train.astype(Xdtype)
#             X_valid = X_valid.astype(Xdtype)
#         if ydtype is not None:
#             y_train = y_train.astype(ydtype)
#             y_valid = y_valid.astype(ydtype)
#         if verbose:
#             print('X_train:', X_train.shape)
#             print('y_train:', y_train.shape)
#             print('X_valid:', X_valid.shape)
#             print('y_valid:', y_valid.shape, '\n')
#         return X_train, y_train, X_valid, y_valid
#     else:
#         X = np.load(f'{full_tgt_dir}/X.npy', mmap_mode=mmap_mode)
#         y = np.load(f'{full_tgt_dir}/y.npy', mmap_mode=mmap_mode)
#         splits = get_predefined_splits(X_train, X_valid)
#         if Xdtype is not None:
#             X = X.astype(Xdtype)
#         if verbose:
#             print('X      :', X.shape)
#             print('y      :', y.shape)
#             print('splits :', coll_repr(splits[0]), coll_repr(splits[1]), '\n')
#         return X, y, splits
#
# def get_Monash_regression_data(dsid, path='./data/Monash', on_disk=True, mode='c', Xdtype='float32', ydtype=None, split_data=True, force_download=False,
#                                verbose=False, timeout=4):
#
#     dsid_list = [rd for rd in Monash_regression_list if rd.lower() == dsid.lower()]
#     assert len(dsid_list) > 0, f'{dsid} is not a Monash dataset'
#     dsid = dsid_list[0]
#     full_tgt_dir = Path(path)/dsid
#     pv(f'Dataset: {dsid}', verbose)
#
#     if force_download or not all([os.path.isfile(f'{path}/{dsid}/{fn}.npy') for fn in ['X_train', 'X_valid', 'y_train', 'y_valid', 'X', 'y']]):
#         if dsid == 'AppliancesEnergy': dset_id = 3902637
#         elif dsid == 'HouseholdPowerConsumption1': dset_id = 3902704
#         elif dsid == 'HouseholdPowerConsumption2': dset_id = 3902706
#         elif dsid == 'BenzeneConcentration': dset_id = 3902673
#         elif dsid == 'BeijingPM25Quality': dset_id = 3902671
#         elif dsid == 'BeijingPM10Quality': dset_id = 3902667
#         elif dsid == 'LiveFuelMoistureContent': dset_id = 3902716
#         elif dsid == 'FloodModeling1': dset_id = 3902694
#         elif dsid == 'FloodModeling2': dset_id = 3902696
#         elif dsid == 'FloodModeling3': dset_id = 3902698
#         elif dsid == 'AustraliaRainfall': dset_id = 3902654
#         elif dsid == 'PPGDalia': dset_id = 3902728
#         elif dsid == 'IEEEPPG': dset_id = 3902710
#         elif dsid == 'BIDMCRR' or dsid == 'BIDM32CRR': dset_id = 3902685
#         elif dsid == 'BIDMCHR' or dsid == 'BIDM32CHR': dset_id = 3902676
#         elif dsid == 'BIDMCSpO2' or dsid == 'BIDM32CSpO2': dset_id = 3902688
#         elif dsid == 'NewsHeadlineSentiment': dset_id = 3902718
#         elif dsid == 'NewsTitleSentiment': dset_id= 3902726
#         elif dsid == 'Covid3Month': dset_id = 3902690
#
#         for split in ['TRAIN', 'TEST']:
#             url = f"https://zenodo.org/record/{dset_id}/files/{dsid}_{split}.ts"
#             fname = Path(path)/f'{dsid}/{dsid}_{split}.ts'
#             pv('downloading data...', verbose)
#             try:
#                 download_data(url, fname, c_key='archive', force_download=force_download, timeout=timeout)
#             except Exception as inst:
#                 print(inst)
#                 warnings.warn(f'Cannot download {dsid} dataset')
#                 if split_data: return None, None, None, None
#                 else: return None, None, None
#             pv('...download complete', verbose)
#             try:
#                 if split == 'TRAIN':
#                     X_train, y_train = _ts2dfV2(fname)
#                     X_train = _check_X(X_train)
#                 else:
#                     X_valid, y_valid = _ts2dfV2(fname)
#                     X_valid = _check_X(X_valid)
#             except Exception as inst:
#                 print(inst)
#                 warnings.warn(f'Cannot create numpy arrays for {dsid} dataset')
#                 if split_data: return None, None, None, None
#                 else: return None, None, None
#         np.save(f'{full_tgt_dir}/X_train.npy', X_train)
#         np.save(f'{full_tgt_dir}/y_train.npy', y_train)
#         np.save(f'{full_tgt_dir}/X_valid.npy', X_valid)
#         np.save(f'{full_tgt_dir}/y_valid.npy', y_valid)
#         np.save(f'{full_tgt_dir}/X.npy', concat(X_train, X_valid))
#         np.save(f'{full_tgt_dir}/y.npy', concat(y_train, y_valid))
#         del X_train, X_valid, y_train, y_valid
#         delete_all_in_dir(full_tgt_dir, exception='.npy')
#         pv('...numpy arrays correctly saved', verbose)
#
#     mmap_mode = mode if on_disk else None
#     X_train = np.load(f'{full_tgt_dir}/X_train.npy', mmap_mode=mmap_mode)
#     y_train = np.load(f'{full_tgt_dir}/y_train.npy', mmap_mode=mmap_mode)
#     X_valid = np.load(f'{full_tgt_dir}/X_valid.npy', mmap_mode=mmap_mode)
#     y_valid = np.load(f'{full_tgt_dir}/y_valid.npy', mmap_mode=mmap_mode)
#     if Xdtype is not None:
#         X_train = X_train.astype(Xdtype)
#         X_valid = X_valid.astype(Xdtype)
#     if ydtype is not None:
#         y_train = y_train.astype(ydtype)
#         y_valid = y_valid.astype(ydtype)
#
#     if split_data:
#         if verbose:
#             print('X_train:', X_train.shape)
#             print('y_train:', y_train.shape)
#             print('X_valid:', X_valid.shape)
#             print('y_valid:', y_valid.shape, '\n')
#         return X_train, y_train, X_valid, y_valid
#     else:
#         X = np.load(f'{full_tgt_dir}/X.npy', mmap_mode=mmap_mode)
#         y = np.load(f'{full_tgt_dir}/y.npy', mmap_mode=mmap_mode)
#         splits = get_predefined_splits(X_train, X_valid)
#         if verbose:
#             print('X      :', X .shape)
#             print('y      :', y .shape)
#             print('splits :', coll_repr(splits[0]), coll_repr(splits[1]), '\n')
#         return X, y, splits