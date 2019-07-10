from __future__ import print_function
import os
import sys
import glob
import numpy
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table, Column
from astropy.units import Quantity
from termcolor import colored
from . import utils as cutils
#import plots
import IPython
import random

try:
    from lsst.afw import image as afwimage
    from lsst.afw import table as afwtable
    import lsst.daf.persistence as dafPersist
except ImportError:
    print(colored("WARNING: LSST stack is probably not installed", "yellow"))

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class DRPLoader(object):

    """Load an LSST DRP output and a few useful things."""
    
    def __init__(self, drp_path,reducedDatasets=None):
        """The 'drp_path' input is the path to the DRP output directory."""

        # Load the bulter
        self.drp_path = drp_path
        self.butler = dafPersist.Butler(self.drp_path)
        self.mapper = self.butler._getDefaultMapper()
        self.repoData = self.butler._repos.outputs()[0]
        
        # Load some basic info on the current DRP
        self.repo_input = self._get_repo("input")    # repo containing the raw data after ingestion
        self.repo_output = self._get_repo("output")  # repo containing the processed data

        # Load some dataids
        self.datasetTypes = self._get_datasetTypes()
        self.datasetTypes_filename = self._get_datasetTypes_withfiles()
        self.dataIds = {}
        defaultDatasets=['raw', 'forced_src', 'deepCoadd_meas', 'deepCoadd_forced_src', 'calexp']   #, 'eimage']:
        if reducedDatasets: defaultDatasets=reducedDatasets[:]
        for dataset in defaultDatasets:
            dataids = self.get_dataIds(dataset)
            if len(dataids):
                self.dataIds[dataset] = dataids

        # Load filter and visits
        self.filters = self.get_filter_list()
        self.visits = self.get_visit_list()

        # Skymap if any
        if self.butler._locate('deepCoadd_skyMap', dafPersist.DataId({}), write=False) is not None:
            self.skymap = self.butler.get("deepCoadd_skyMap")
            self.skymap_name = self.skymap.__class__.__name__
            self.skymap_doc = self.skymap.__doc__
            self.skymap_config = self.skymap.config.toDict()
            self.skymap_numtracts = self.skymap._numTracts
            self.skymap_numpatches = self.skymap[0].getNumPatches()

        # Mapper info
        self.mapper_name = self.mapper.__name__
        self.mapper_package = self.mapper.packageName 
        self.mapper_camera = self.mapper.getCameraName()

        # Packages
        self.packages = self.butler.get('packages')

        # Other
        self.configs = self._load_configs()
        self.schemas = self._load_schemas()

    def _get_repo(self, repo):
        """Get the full path of the input/output repository."""
        has_parent = bool(len(self.repoData.getParentRepoDatas()))
        if repo == 'output':
            # The given repo is an output one if it has a parent, otherwise, it should be an input one
            return self.repoData.cfgRoot if has_parent else 'No output directory found'
        elif repo == 'input':
            # the parent directory is the one containing the raw data (after ingestion)
            if has_parent:
                parentRepoData = self.repoData.getParentRepoDatas()[0]         
                return os.path.realpath(parentRepoData.cfgRoot)  # input path -> ../input in this case
            else:
                return self.repoData.cfgRoot
        else:
            raise IOError("Wrong repo name. You should not be calling this internal method anyway.")

    def _get_datasetTypes(self):
        return sorted(self.repoData.repo._mapper.mappings.keys())

    def _get_datasetTypes_withfiles(self):
        mappables = [m for m in dir(self.repoData.repo._mapper) if m.startswith('map_')]
        withfile = [m.replace('map_', '') for m in mappables if m.endswith('_filename')]
        return sorted(withfile)

    def get_catIdKeys(self, datasetType):
        """Get the list of ID keys for a given catalog."""
        if datasetType not in self.datasetTypes:
            msg = "%s is not a valid datasetType. Check self.datasetTypes for the valid list." % \
                  datasetType
            raise IOError(msg)
        return self.butler.getKeys(datasetType)

    def get_dataIds(self, datasetType):
        """Get all available data id for a given dataType."""
        keys = self.get_catIdKeys(datasetType)
        # 'tract' is present in the keys for the forced_src cat, but should not be
        if datasetType == 'forced_src':
            del keys['tract']
        try:
            metadata = self.butler.queryMetadata(datasetType, format=sorted(keys.keys()))
        except:
            metadata = None
        if metadata is not None:
            return [dict(zip(sorted(keys.keys()), list(v) if not isinstance(v, list) else v))
                    for v in metadata]
        else:
            if datasetType not in self.repoData.repo._mapper.datasets:
                return []
            template = self.repoData.repo._mapper.datasets[datasetType]._template
            path = os.path.join(self.repoData.cfgRoot, os.path.dirname(template))
            basepath = "/".join([p for p in path.split('/') if not p.startswith('%')]) + "/"
            keys = self.butler.getKeys(datasetType)
            gpath = "/".join([p if not p.startswith('%') else '*' for p in path.split('/')])
            paths = [p for p in glob.glob(gpath) if 'merged' not in p]
            return [{k: keys[k](v) for k, v in zip(keys, p.split(basepath)[1].split('/'))}
                    for p in paths]

    def get_filter_list(self):
        """Get the list of filters."""
        return set([dataid['filter'] for dataid in self.dataIds['raw']])

    def get_tract_list(self):
        """Get the list of tracts."""
        keyList=dataid.keys()
        keyList.sort()
        return set([dataid['tract'] for dataid in self.dataIds['raw']])

    def get_patch_list(self):
        """Get the list of patches."""
        return set([dataid['patch'] for dataid in self.dataIds['raw']])

    def get_visit_list(self):
        """"All available vists."""
        visits = {filt: list(set([dataid['visit'] 
                                  for dataid in self.dataIds['raw'] if dataid['filter'] == filt]))
                  for filt in self.filters}
        return visits

    def _load_configs(self):
        """Load configs for the main tasks."""
        configs = self._load_generic_dataset("config")
        return {cfg: configs[cfg].toDict() for cfg in configs}

    def _load_schemas(self):
        """Load the schemas for all catalogs."""
        schemas = self._load_generic_dataset("schema")
        for schema in schemas:
            sch = schemas[schema].asAstropy()
            schemas[schema] = {col: {'description': sch[col].description, 
                                     'unit': sch[col].unit,
                                     'dtype': sch[col].dtype}
                               for col in sch.colnames}
        return schemas
        
    def _load_generic_dataset(self, datatype):
        """Load the schema or config datasets."""
        if datatype not in ['config', 'schema']:
            raise IOError("`datatype` must be either `config` or `schema`.")
        datatypes = {}
        for dataset in self.datasetTypes:
            if not dataset.endswith('_%s' % datatype):
                continue
            for dataId in ([{}] + [self.dataIds[key][0] for key in self.dataIds]):
                try:
                    datatypes[dataset] = self.butler.get(dataset, dataId=dataId)
                    break
                except:
                    pass
        return datatypes
    
    def overview(self):
        """Overview of the current DRP content."""
        # General info on the data repository
        html = "<h2>General info on the current DRP</h2>"
        
        # Repos
        html += "<h3>Paths to the repositories</h3>"
        html += cutils.square_list(["<b>Input</b>: %s</li>" % self.repo_input,
                                    "<b>Output</b>: %s</li>" % self.repo_output
                                   ]
                                  )

        # Info on the mapper, camera, package
        html += "<h3>Mapper info</h3>"
        html += cutils.square_list(["<b>Package</b>: %s" % self.mapper_package,
                                     "<b>Camera</b>: %s" % self.mapper_camera,
                                     "<b>Name</b>: %s" % self.mapper_name
                                    ]
                                   )
                                   
            
        html += "<h3>Filters and visits</h3>"
        html += "<table>"
        html += "<tr><th>Name</th>"
        html += "".join(["<td>%s</td>" % filt for filt in self.filters])
        html += "</tr>"
        html += "<tr><th>#Visits</th>"
        html += "".join(["<td>%i</td>" % len(self.visits[filt]) for filt in self.filters])
        html += "</tr>"
        html += "</table>"
        
        # Other info, filter, skymap, etc.
        items = []
        if hasattr(self, 'skymap'):
            items.append("<b>Sky map</b>: %s" % str(self.skymap_name))
        if len(items):
            html += "<h3>Other info</h3>"
            html += cutils.square_list(items)

        return IPython.display.HTML(html)


class DRPImages(DRPLoader):

    def __init__(self, drp_path,reducedDatasets=None):
        """The 'drp_path' input is the path to the DRP output directory."""
        super().__init__(drp_path,reducedDataset)

    def get_file(self, datatype, dataid):
        try:
            cfiles = self.butler.get('%s_filename' % datatype, dataId=dataid)
            for i, cfile in enumerate(cfiles):
                if self.repo_output in cfile and not os.path.exists(cfile):
                    cfiles[i] = cfile.replace(self.repo_output, self.repo_input)
            return cfiles
        except:
            return []

    def get_files(self, datatype, filt=None, visit=None, tract=None, patch=None):
        dataids = self.get_dataid_from_dataset(datatype)
        files = numpy.concatenate([self.get_file(datatype, dataid) for dataid in dataids])
        return files

    def get_dataid_from_dataset(self, datatype, test=False):
        try:
            keys = self.butler.getKeys(datatype)
        except:
            keys = {}
        if not len(keys):
            return [{}]
        elif 'visit' in keys and 'tract' in keys:
            key = 'forced_src'
            dataIds = [self.dataIds[key][0]] if test else self.dataIds[key]
        elif 'visit' in keys:
            key = 'raw'
            dataIds = [self.dataIds[key][0]] if test else self.dataIds[key]
        elif 'tract' in keys:
            key = 'deepCoadd_meas'
            dataIds = [self.dataIds[key][0]] if test else self.dataIds[key]
        else:
            dataIds = self.get_dataIds(datatype)
            if not len(dataIds):
                dataIds = [{}]
        return [{k: dataid[k] for k in dataid if k in keys} for dataid in dataIds]

    def _has_file(self, datatype):
        return bool(len(self.get_file(datatype, self.get_dataid_from_dataset(datatype, test=True)[0])))

    def _type_file(self, datatype):
        return self.get_file(datatype, self.get_dataid_from_dataset(datatype, test=True)[0])
    
    def display(self, datatype, dataid, display='matplotlib'):
        if display == 'matplotlib':
            image = self.butler.get(datatype, dataid)
            plots.display_matplotlib(image)


class DRPCatalogs(DRPLoader):

    """Load catalogs from an LSST DRP output path."""

    def __init__(self, drp_path,reducedDatasets=None):
        """The 'drp_path' input is the path to the DRP output directory."""
        super().__init__(drp_path,reducedDatasets)

        # Initialize data dictionnaries
        self.catalogs = {}
        self.keys = {}
        self.missing = {}
        self.from_butler = {'getmag': None, 'wcs': None,
                            'schema': None, 'extension': None}
        self.append = False

    def _load_catalog_dataid(self, catalog, dataid, table=True, **kwargs):
        """Load a catalog from a 'dataId' set of parameter."""
        try:
            cat = self.butler.get(catalog, dataId=dataid,
                                  flags=afwtable.SOURCE_IO_NO_FOOTPRINTS)
        except:  # OperationalError: no such column: flags
            cat = self.butler.get(catalog, dataId=dataid)
        if self.from_butler['schema'] is None and hasattr(cat, 'getSchema'):
            self.from_butler['schema'] = cat.getSchema()
        return cat.getColumnView().extract(*self.keys[catalog],
                                           copy=True, ordered=True) if table else cat

    def _get_catalog(self, dataset, **kwargs):
        """Load the catalogs from the butler."""
        filenames = (self.butler.get(dataset + "_filename",
                                     dataId, immediate=True)[0]
                     for dataId in self.dataIds[dataset])
        try:  # In recent stack version, metadata are in HDU 1
            headers = (afwimage.readMetadata(fn, 1) for fn in filenames)
            size = sum(md.get("NAXIS2") for md in headers)
        except:  # Older stack version
            headers = (afwimage.readMetadata(fn, 2) for fn in filenames)
            size = sum(md.get("NAXIS2") for md in headers)
        cat = self.butler.get(dataset, self.dataIds[dataset][0],
                              flags=afwtable.SOURCE_IO_NO_FOOTPRINTS, immediate=True)
        self.from_butler['schema'] = cat.schema
        catadic = {k: [] for k in sorted(self.dataIds[dataset][0].keys())}
        catalog = afwtable.SourceCatalog(self.from_butler['schema'])
        catalog.reserve(size)
#        pbar = cutils.progressbar(len(self.dataIds[dataset]))
        print("INFO: Looping over the dataids")
        for i, dataid in enumerate(self.dataIds[dataset]):
            cat = self.butler.get(dataset, dataid,
                                  flags=afwtable.SOURCE_IO_NO_FOOTPRINTS)
            catalog.extend(cat, deep=True)
            for newkey in catadic:
                catadic[newkey].extend([dataid[newkey]] * len(cat))
#            pbar.update(i + 1)
#        pbar.finish()
        print("INFO: Merging the dictionnaries")
        catadic.update(catalog.getColumnView().extract(*self.keys[dataset],
                                                       copy=True, ordered=True))
        return catadic


    def _get_catalog_MT_thread(self, args):

        iThread,dataset,dataId=args
        cat=None
        try: 
            cat = self.butler.get(dataset, dataId,
                                  flags=afwtable.SOURCE_IO_NO_FOOTPRINTS)
        except:
            print("\n\n\nERROR : while reading %s %s file\n\n\n"%(dataset,str(dataId)))
            return dataId,None
        
        return dataId,cat


    def _get_catalog_MT(self, dataset, **kwargs):
        """Load the catalogs from the butler using multi threaded python."""

        # Get corresponding filenames & dataIds
        inputData = tuple((self.butler.get(dataset + "_filename",
                                          dataId, immediate=True)[0],dataId)
                          for dataId in self.dataIds[dataset])
        # Check if filenames exist ( issues with deepCoadd_forced_src non existing files)
        filesInError=[]
        for fn,dataId in inputData:
            if not os.path.isfile(fn):
                filesInError.append((fn,dataId))
        print("File number : ",len(inputData),"  // files in error : ",filesInError)

        filenames=[fn for fn,tmp in inputData]
        local_dataIds=self.dataIds[dataset][:]
        for fn,dataId in filesInError:
            local_dataIds.remove(dataId)
            filenames.remove(fn)

        print("File number - final list : ",len(local_dataIds))

        # Loop over filenames to read metaData
        try:  # In recent stack version, metadata are in HDU 1
            headers = (afwimage.readMetadata(fn, 1) for fn in filenames)
            size = sum(md.get("NAXIS2") for md in headers)
        except:  # Older stack version
            headers = (afwimage.readMetadata(fn, 2) for fn in filenames)
            size = sum(md.get("NAXIS2") for md in headers)


#        cat = self.butler.get(dataset, self.dataIds[dataset][0],
#                              flags=afwtable.SOURCE_IO_NO_FOOTPRINTS, immediate=True)
        cat = self.butler.get(dataset, local_dataIds[0],
                              flags=afwtable.SOURCE_IO_NO_FOOTPRINTS, immediate=True)
        self.from_butler['schema'] = cat.schema
#        catadic = {k: [] for k in sorted(self.dataIds[dataset][0].keys())}
        catadic = {k: [] for k in sorted(local_dataIds[0].keys())}
        catalog = afwtable.SourceCatalog(self.from_butler['schema'])
        catalog.reserve(size)

        # MultiThreading process parameters - as defined in kwargs
        nbMaxSimultaneousThread=4 
#        nbMaxDataIdsPerPool=10
        if "MT_MaxThread" in kwargs: nbMaxSimultaneousThread = kwargs["MT_MaxThread"]
#        if "MT_DataIdsPerThread" in kwargs: nbMaxDataIdsPerPool = kwargs["MT_DataIdsPerThread"]

#        stepPoolList=[(i,min(i+nbMaxIdPerPool,nbIds)) for i in range(0, nbIds, nbMaxIdPerPool)]
#        print(stepPoolList)
        
        time0=time.time()

        icmpt=0
        runningThreads=[]
        nbDataIds = len(local_dataIds)   ##self.dataIds[dataset])
        with ThreadPoolExecutor(max_workers=nbMaxSimultaneousThread) as executor:
            
            for i,dataId in enumerate(local_dataIds):   ##self.dataIds[dataset]):
                arg=(i,dataset,dataId)
                runningThreads.append(executor.submit(self._get_catalog_MT_thread, arg))
                            
            for x in as_completed(runningThreads):

                dataid,cat = x.result()

                if cat:
                    catalog.extend(cat, deep=True)

                    for newkey in catadic:
                        catadic[newkey].extend([dataid[newkey]] * len(cat))
                        
                progress=int(icmpt*100./nbDataIds)
#                sys.stdout.write("Download progress: %d%s   - %d/%d\r" % (progress,"%",icmpt,nbDataIds))
#                sys.stdout.flush()
                print("Download progress: %d%s   - %d/%d" % (progress,"%",icmpt,nbDataIds))
                icmpt+=1
        print("\n")
        
        time1=time.time()
        print("INFO: Catalogs reading done - %f s"%(time1-time0))

        print("INFO: Merging the dictionnaries")
        catadic.update(catalog.getColumnView().extract(*self.keys[dataset],
                                                       copy=True, ordered=True))
        return catadic


    def _load_catalog(self, catalog, **kwargs):
        """Load a given catalog."""

        if len(self.dataIds[catalog])==0: return

        if "MT" in kwargs and kwargs["MT"]:
            print("INFO: Getting the data from the butler for %i fits files - multi-thread process" % len(self.dataIds[catalog]))
            self.catalogs[catalog] = Table(self._get_catalog_MT(catalog, **kwargs))
        else:
            print("INFO: Getting the data from the butler for %i fits files" %
                  len(self.dataIds[catalog]))
            self.catalogs[catalog] = Table(self._get_catalog(catalog, **kwargs))
        print("INFO: Getting descriptions and units")
        for k in self.catalogs[catalog].keys():
            if k in self.from_butler['schema']:
                asfield = self.from_butler['schema'][k].asField()
                self.catalogs[catalog][k].description = cutils.shorten(asfield.getDoc())
                self.catalogs[catalog][k].unit = asfield.getUnits()
        self.from_butler['schema'] = None
        print("INFO: %s catalog loaded (%i sources)" %
              (catalog, len(self.catalogs[catalog])))
        self._add_new_columns(catalog)
        if 'matchid' in kwargs and catalog == 'forced_src':
            self._match_ids()
        if 'output_name' in kwargs:
            self.save_catalogs(kwargs['output_name'], catalog,
                               kwargs.get('overwrite', False), delete_catalog=True)

    def _match_deepcoadd_catalogs(self):
        """In case of missing data for one catalog, remove corresonding data from the other."""
        if 'deepCoadd_meas' in self.catalogs and 'deepCoadd_forced_src' in self.catalogs:
            if len(self.catalogs['deepCoadd_meas']) == len(self.catalogs['deepCoadd_forced_src']):
                return
            print(colored("\nINFO: matching 'deepCoadd_meas' and 'deepCoadd_forced_src' catalogs",
                          'green'))
            if 'deepCoadd_meas' in self.missing:
                for dataid in self.missing['deepCoadd_meas']:
                    filt = (self.catalogs['deepCoadd_forced_src']['filter'] == dataid['filter']) & \
                           (self.catalogs['deepCoadd_forced_src']
                            ['patch'] == dataid['patch'])
                    self.catalogs['deepCoadd_forced_src'] = self.catalogs['deepCoadd_forced_src'][~filt]
            if 'deepCoadd_forced_src' in self.missing:
                for dataid in self.missing['deepCoadd_forced_src']:
                    filt = (self.catalogs['deepCoadd_meas']['filter'] == dataid['filter']) & \
                           (self.catalogs['deepCoadd_meas']
                            ['patch'] == dataid['patch'])
                    self.catalogs['deepCoadd_meas'] = self.catalogs['deepCoadd_meas'][~filt]

    def _match_ids(self):
        """Select in the 'forced_src' catalog the source that are in the deepCoad catalogs."""
        deepcoadd = [cat for cat in self.catalogs if 'deepCoadd' in cat]
        if len(deepcoadd):
            if 'forced_src' in self.catalogs:
                print(
                    colored("\nINFO: Matching 'forced_src' and 'deepCoadd' catalogs", "green"))
                print("  - %i sources in the forced-src catalog before selection" %
                      len(self.catalogs['forced_src']))
                coaddid = 'id' if 'id' in self.catalogs[deepcoadd[0]].keys(
                ) else 'objectId'
                filt = numpy.where(numpy.in1d(self.catalogs['forced_src']['objectId'],
                                        self.catalogs[deepcoadd[0]][coaddid]))[0]
                self.catalogs['forced_src'] = self.catalogs['forced_src'][filt]
                print("  - %i sources in the forced-src catalog after selection" %
                      len(self.catalogs['forced_src']))
            else:
                print(colored("\nWARNING: forced_src catalogs not loaded. No match possible.",
                              "yellow"))
        else:
            print(colored("\nWARNING: No deepCoadd* catalog loaded. No match possible.",
                          "yellow"))

    def _add_new_columns(self, catalog=None):
        """Compute magns for all fluxes of a given table. Add the corresponding new columns.

        Compute the x/y position in pixel for all sources. Add new columns to the table.
        """
        print(colored("\nINFO: Adding magnitude and coordinates columns", "green"))
        catalogs = [catalog] if catalog is not None else list(self.catalogs)
        for catalog in catalogs:
            # skip wcs key
            if catalog == 'wcs':
                continue
            print("  - for", catalog)
            columns = []
            # Add magnitudes
            if self.from_butler['getmag'] is not None:
                kfluxes = [
                    k for k in self.catalogs[catalog].columns if k.endswith('_flux')]
                ksigmas = [k + 'Sigma' for k in kfluxes]
                print("    -> getting magnitudes")

                for kflux, ksigma in zip(kfluxes, ksigmas):
                    if kflux.replace('_flux', '_mag') in self.catalogs[catalog].keys():
                        continue

                    if ksigma in self.catalogs[catalog].keys():
                        mag, dmag = self.from_butler['getmag'](numpy.array(self.catalogs[catalog][kflux],
                                                                        dtype='float'),
                                                               numpy.array(self.catalogs[catalog][ksigma],
                                                                        dtype='float'))

                        print("-> new param : magnitudes : ",catalog," add param ",kflux.replace('_flux', '_mag'))
                        print("-> new param : magnitudes : ",catalog," add param ",ksigma.replace('_flux', '_mag'))

                        columns.append(Column(name=kflux.replace('_flux', '_mag'),
                                          data=mag, description='Magnitude', unit='mag'))
                   
                        columns.append(Column(name=ksigma.replace('_fluxSigma', '_magSigma'),
                                          data=dmag, description='Magnitude error', unit='mag'))
            if 'x_Src' in self.catalogs[catalog].keys():
                return
            ra = Quantity(self.catalogs[catalog]["coord_ra"].tolist(), 'rad')
            dec = Quantity(self.catalogs[catalog]["coord_dec"].tolist(), 'rad')
            # Get the x / y position in pixel
            if self.from_butler['wcs'] is not None:
                print("    -> getting pixel coordinates")
                xsrc, ysrc = SkyCoord(ra, dec).to_pixel(
                    self.from_butler['wcs'])

                print("-> new param : pixel coord : ",catalog," add param x_Src uSrc")

                columns.append(Column(name='x_Src', data=xsrc,
                                      description='x coordinate', unit='pixel'))
                columns.append(Column(name='y_Src', data=ysrc,
                                      description='y coordinate', unit='pixel'))
            else:
                print(colored("\nWARNING: no WCS found for this dataset", "yellow"))

            # Get coordinates in degree
            print("    -> getting degree coordinates")
            print("-> new param : deg coord : ",catalog," add coord_ra_deg coord_dec_deg")
            columns.append(Column(name='coord_ra_deg', data=Angle(ra).degree,
                                  description='RA coordinate', unit='degree'))
            columns.append(Column(name='coord_dec_deg', data=Angle(dec).degree,
                                  description='DEC coordinate', unit='degree'))

            # Adding all new columns
            print("    -> adding all the new columns")
            self.catalogs[catalog].add_columns(columns)
            # Clean memory before going further
            # gc.collect()

    def _load_calexp(self, calcat='deepCoadd_calexp', **kwargs):
        """Load the deepCoadd_calexp info in order to get the WCS and the magnitudes."""

        print(colored("\nINFO: Loading the %s info" % calcat, 'green'))
        print("INFO: Getting the %s catalog for one dataId" % calcat)
        self.dataIds[calcat] = self.dataIds['deepCoadd_meas']

        if len(self.dataIds[calcat])==0: return

        calexp = self._load_catalog_dataid(
            calcat, self.dataIds[calcat][0], table=False)
        print("INFO: Getting the magnitude function")
        calib = calexp.getPhotoCalib()
#        calib = calexp.getCalib()
#        calib.setThrowOnNegativeFlux(False)
        self.from_butler['getmag'] = calib.getMagnitude
        print("INFO: Getting the wcs function")
        wcs = calexp.getWcs().getFitsMetadata().toDict()
        self.from_butler['wcs'] = WCS(wcs)
        self.catalogs['wcs'] = Table({k: [wcs[k]] for k in wcs})

    def load_catalogs(self, catalogs, **kwargs):
        """Load a list of catalogs.

        :param str/list catalogs: A catalog name, or a list of catalogs (see below)
        :param dict keys: A dictionnary of keys to load for each catalog

        Available kwargs are:

        :param bool update: Set to True if you want to update an already loaded catalog
        :param bool show: Set to True to get all available keys of a (list of) catalog(s)
        :param bool matchid: Will only keep objects which are in the deepCoad catalogs (to be used
                             when loading the forced_src and deepCoadd catalogs)
        :param list filter : Set of filter to be read 
        :param list tract : Set of filter to be read 
        :param list patch : Set of filter to be read 

        Examples of catalogs that you can load:

         - 'deepCoadd_ref',
         - 'deepCoadd_meas',
         - 'deepCoadd_forced_src',
         - 'deepCoadd_calexp',
         - 'forced_src'
         - 'src'
        """
        if 'show' in kwargs:
            self.show_keys(catalogs)
            return

        # Check if the complete set of dataIds are to be read
        #     or only partially depending on filter/tract/patch lists defined in kwargs
        for cat in self.dataIds:

            print("Init dataIds : ",cat," ",len(self.dataIds[cat]))
#dataIds[cat][0])
#            for key, value in kwargs.items():
#                print("%s - %s"%(key,value))
            
            keyList=[x for x in ['filter','tract','patch'] if x in self.dataIds[cat][0]]
#            print("key list : ",keyList)
            keyList_kwargs=[x for x in keyList if kwargs and x in kwargs]
#            print("key list kwargs : ",keyList_kwargs)

            for key in keyList_kwargs: print("kwargs : ",kwargs[key])

            dataIds_new=[]
            if len(keyList_kwargs)>0:
                for dataId in self.dataIds[cat]:
                    bMatch=True
                    for key in keyList:
                        if not dataId[key] in kwargs[key]: bMatch=False
                    if bMatch: dataIds_new.append(dataId)

                self.dataIds[cat]=dataIds_new[:]

            print("Remaining dataIds : ",cat," ",len(self.dataIds[cat]))

        # One dataId only (used mainly to retrieve catalog data/typestructure)
        if "oneIdOnly" in kwargs and kwargs["oneIdOnly"]:
            for i,cat in enumerate(self.dataIds):
                print("CAT ",cat,"\n",self.dataIds[cat][0])                
            idx=0
            for i,cat in enumerate(self.dataIds):
#                if i==0: idx=random.randrange(0,len(self.dataIds[cat]))
                self.dataIds[cat]=self.dataIds[cat][idx:idx+1]


        # Further kwargs deconding
        keys = {} if 'keys' not in kwargs else kwargs['keys']
        catalogs = [catalogs] if isinstance(catalogs, str) else catalogs

        print("---------------------------------------------------")
        print("catalogs : ",catalogs)
        print("---------------------------------------------------")

        # Read calexp
        if any(["deepCoadd" in cat for cat in catalogs]):
            self._load_calexp(**kwargs)
        else:
            self._load_calexp(calcat='calexp', **kwargs)

        # Loop over catalogs - read data
        for catalog in sorted(catalogs):
            if catalog in self.catalogs and 'update' not in kwargs:
                print(colored("\nWARNING: %s is already loaded. Use 'update' to reload it." %
                              catalog, "yellow"))
                continue
            if 'calexp' in catalog:
                print(colored("\nWARNING: Skipping %s. Not a regular catalog (no schema).\n" %
                              catalog, "yellow"))
                continue
            print(colored("\nINFO: Loading the %s catalog" % catalog, 'green'))
            self.keys[catalog] = keys.get(catalog, "*")
            self._load_catalog(catalog, **kwargs)

        # Check if deepCoadd_meas and forced_src are both defined 
        self._match_deepcoadd_catalogs()

        # Save data
        if 'output_name' in kwargs and self.from_butler['wcs'] is not None:
            self.save_catalogs(kwargs['output_name'],
                               'wcs', kwargs.get('overwrite', False))
        print(colored("\nINFO: Done loading the data.", "green"))

    def show_keys(self, catalogs=None):
        """Show all the available keys."""
        if catalogs is None:
            catalogs = [k for k in self.catalogs.keys() if k != 'wcs']
        catalogs = [catalogs] if isinstance(catalogs, str) else catalogs
        if len(catalogs) == 0:
            print(colored("\nWARNING: No catalog loaded nor given.", "yellow"))
            return
        for cat in catalogs:
            if cat not in self.dataIds:
                print(colored("\nINFO: Get the available data IDs", "green"))
            print(colored("\nINFO: Available list of keys for the %s catalog" % cat, "green"))
            schema = list(self.schemas[cat + '_schema'])
            print("  -> %i keys available for %s" % (len(schema), cat))
            print("  -> All saved in %s_keys.txt" % cat)
            numpy.savetxt("%s_keys.txt" % cat, schema, fmt="%s")

    def save_catalogs(self, output_name, catalog=None, overwrite=False, delete_catalog=False):
        """Save the catalogs into an hdf5 file."""
        if not output_name.endswith('.hdf5'):
            output_name += '.hdf5'
        print(colored("\nINFO: Saving the catalogs in %s" % output_name, "green"))
        catalogs = [catalog] if catalog is not None else self.catalogs
        
        for cat in catalogs:
            print("  - saving", cat)
            for k in self.catalogs[cat].keys():
                if isinstance(self.catalogs[cat][k][0], str):
                    self.catalogs[cat].replace_column(
                        k, Column(self.catalogs[cat][k].astype('bytes')))
            if not self.append:
                self.catalogs[cat].write(output_name, path=cat, compression=True,
                                         serialize_meta=True, overwrite=overwrite)
            else:
                self.catalogs[cat].write(output_name, path=cat, compression=True,
                                         serialize_meta=True, append=True)
            if delete_catalog and cat is not 'wcs':
                oid = self.catalogs[cat]['id' if 'id' in self.catalogs[cat].keys()
                                         else 'objectId'].copy()
                self.catalogs.pop(cat)
                self.catalogs[cat] = Table([oid]).copy()
            self.append = True
        print("INFO: Saving done!")
        # Clean memory before loading a new catalog
        # gc.collect()


class QservNameConverter(object):

    qserv_shortNameKeys_prefix=[

##     ("ext_shapeHSM_HsmPsfMoments","esHSM_HPM"),
##     ("ext_shapeHSM_HsmShapeRegauss","esHSM_HSRgaus"),
##     ("ext_shapeHSM_HsmSourceMoments","esHSM_HSMs"),
##     ("ext_shapeHSM_HsmPsfMoments","esHSM_HPMs"),
##     ("ext_shapeHSM_HsmShapeLinear","esHSM_HSLr"),
##     ("ext_shapeHSM_HsmShapeBj","esHSM_HSBj"),
##     ("ext_shapeHSM_HsmShapeKsb","esHSM_HSK"),

    ("ext_shapeHSM_HsmPsfMoments","HPM"),
    ("ext_shapeHSM_HsmShapeRegauss","HSRgaus"),
    ("ext_shapeHSM_HsmSourceMoments","HSMs"),
    ("ext_shapeHSM_HsmPsfMoments","HPMs"),
    ("ext_shapeHSM_HsmShapeLinear","HSLr"),
    ("ext_shapeHSM_HsmShapeBj","HSBj"),
    ("ext_shapeHSM_HsmShapeKsb","HSK"),

    ("ext_convolved_ConvolvedFlux","eCF"),

    ("merge_peak_","mp_"),
    ("merge_footprint_","mf_"),
    ("deblend_","dbd_"),

    ("base_PixelFlags_flag","bPFf"),
    ("base_ClassificationExtendedness","bClEx"),
    ("base_CircularApertureFlux","bCAF"),
    ("base_SdssShape_flag","bSSfg"),
    ("base_SdssShape_flux","bSSfx"),
    ("base_SdssShape_x","bSSx"),
    ("base_SdssShape_y","bSSy"),
    ("base_SdssShape_psf","bSSpsf"),
    ("base_SdssCentroid","bSSCt"),
    ("base_InputCount","bIC"),
    ("base_GaussianFlux","bGaF"),
    ("base_PsfFlux","bPsF"),
    ("base_GaussianCentroid","bGC"),
    ("base_NaiveCentroid","bNC"),
    ("base_Variance","bVr"),
    ("base_LocalBackground","bLBG"),
    ("base_Blendedness_raw_parent","bBrp"),
    ("base_Blendedness_raw_child","bBrc"),
    ("base_Blendedness_raw_flux","bBrf"),
    ("base_Blendedness_abs_parent","bBap"),
    ("base_Blendedness_abs_child","bBac"),
    ("base_Blendedness_abs_flux","bBaf"),
    ("base_Blendedness_flag","bBflg"),

    ("modelfit_CModel_initial_","mCMi_"),
    ("modelfit_CModel_exp_","mCMe_"),
    ("modelfit_CModel_region_","mCMr_"),
    ("modelfit_CModel_dev_","mCMd_"),
    ("modelfit_CModel_flag_","mCM_f_"),
    ("modelfit_CModel_flags_","mCM_fs_"),
    ("modelfit_CModel_flux_inner","mCM_fi"),
    ("modelfit_CModel_fracDev","mCM_fD"),
    ("modelfit_CModel_objective","mCM_ob"),

    ("modelfit_GeneralShapeletPsfApprox_DoubleShapelet","mGSPA_DS"),
    ("modelfit_DoubleShapeletPsfApprox","mDSPApx"),
    ("modelfit_DSPA","mDSPA"),
    ("modelfit_DStPsfApprox","m_DPAx"),

    ("slot_ModelFlux_initial_","sMoFi_"),
    ("slot_ModelFlux_exp_","sMoFe_"),
    ("slot_ModelFlux_region_","sMoFr_"),
    ("slot_ModelFlux_dev_","sMoFd_"),
    ("slot_Centroid_","sCt_"),
    ("slot_ModelFlux_flag_","sMoF_f_"),
    ("slot_ModelFlux_flags_","sMoF_fs_"),

    ("slot_Shape_xx","sSxx"),
    ("slot_Shape_yy","sSyy"),
    ("slot_Shape_x","sSx"),
    ("slot_Shape_y","sSy"),
    ("slot_Shape_xy","sSxy"),
    ("slot_Shape_fg","sSfg"),
    ("slot_Shape_fnpx","sSfnpx"),
    ("slot_Shape_fncn","sSfncn"),
    ("slot_Shape_fpso","sSfpso"),

    ("slot_PsfFlux_","sPF_"),

    ("slot_ModelFlux_","xsMoF_"),
    ("modelfit_CModel_","xmCM_"),

    ]

    qserv_shortNameKeys_suffix=[

    ("flag_badCentroid_noSecondDerivative","fbCnSD"),
    ("flag_badCentroid_almostNoSecondDerivative","fbCaNSD"),
    ("flag_badCentroid_notAtMaximum","fbCnAM"),
    ("flag_badCentroid_resetToPeak","fbCrTP"),
    ("flag_badCentroid_edge","fbCe"),
    ("flag_badCentroid_notAtMaximum","fbCnAMx"),
    ("flag_badCentroid_resetToPeak","fbCrTP"),

    ("flag_badInitialCentroid_noSecondDerivative","fbiCnSD"),
    ("flag_badInitialCentroid_almostNoSecondDerivative","fbiCaNSD"),
    ("flag_badInitialCentroid_notAtMaximum","fbiCnAMx"),
    ("flag_badInitialCentroid_resetToPeak","fbiCrTP"),
    ("flag_badInitialCentroid_edge","fbiCed"),

    ("flag_apertureTruncated","faT"),
    ("flag_sincCoeffsTruncated","fsCT"),        

    ("flag_noSecondDerivative","fnSD"),
    ("flag_almostNoSecondDerivative","faNSD"),

    ("flag_badCentroid","fbCn"),
    ("flag_badInitialCentroid","fbiCn"),

    ("badCentroid_noSecondDerivative","xbCnSD"),
    ("badCentroid_almostNoSecondDerivative","xbCaNSD"),
    ("badCentroid_notAtMaximum","xbCnAMx"),
    ("badCentroid_resetToPeak","xbCrTP"),
    ("badCentroid_edge","xbCed"),

    ("badInitialCentroid_notAtMaximum","xbiCnAMx"),
    ("badInitialCentroid_resetToPeak","xbiCrTP"),

    ("region_usedInitialEllipseMax","rguIEMax"),
    ("region_usedInitialEllipseMin","rguIEMin"),
    ("region_usedFootprintArea","rguFA"),
    ("region_initial_ellipse_xy","rielxy"),
    ("region_initial_ellipse_xx","rielxx"),
    ("region_initial_ellipse_yy","rielyy"),
    ("region_final_ellipse_xy","rfelxy"),
    ("region_final_ellipse_xx","rfelxx"),
    ("region_final_ellipse_yy","rfelyy"),
    ("region_maxBadPixelFraction","rmBPF"),
    ("region_usedPsfArea","ruPS"),
    ("region_maxArea","rmA"),

    ("flag_no_pixels","fnpx"),
    ("flag_not_contained","fncn"),
    ("flag_parent_source","fpso"),
    ("flag_galsim","fgs"),

    ("flag_invalidPointForPsf","fiPFP"),
    ("flag_invalidMoments","fiMt"),
    ("flag_maxIterations","fmxIt"), 

    ("interpolatedCenter","iCt"),
    ("saturatedCenter","sCt"),
    ("crCenter","crCt"),
    ("suspectCenter","suCt"),
    ("clippedCenter","cCt"),

    ("flag_trSmall","ftrS"),
    ("flag_maxIter","fmI"),
    ("nIter","nIr"),
    ("flag_numericError","fnEr"),
    ("flag_noGoodPixels","fnGPx"),
    ("flag_noPsf","fnPf"),
    ("flag_emptyFootprint","feFpt"),
    ("flag_notAtMaximum","fnAM"),
    ("flag_resetToPeak","frTP"),

    ("flag_badShape","fbShp"),
    ("flag_badShape_no_pixels","fbShnp"),
    ("flag_badShape_not_contained","fbShnc"),
    ("flag_badShape_parent_source","fbShps"),
        

    ("_magSigma","_mgSg"),
    ("_fluxSigma","_fxSg"), 

    ("_initial_ellipse_xy","_ielxy"),
    ("_initial_ellipse_xx","_ielxx"),
    ("_initial_ellipse_yy","_ielyy"),
    ("_final_ellipse_xy","_felxy"),
    ("_final_ellipse_xx","_felxx"),
    ("_final_ellipse_yy","_felyy"),
    ("_ellipse_xy","_elxy"),
    ("_ellipse_xx","_elxx"),
    ("_ellipse_yy","_elyy"),
    ("_nonlinear_0","_nl0"),
    ("_nonlinear_1","_nl1"),
    ("_nonlinear_2","_nl2"),
    ("_fixed_0","_fx0"),
    ("_fixed_1","_fx1"),
    ("_flux","_fx"),
    ("_flag","_fg"),
    ("_apCorrSigma","_aCSg"),
    ("_badCentroid","_xbCt"),
    ("flag_apCorr","faCo"),
    ("_apCorr","_aCo"),
    ("_noShapeletPsf","_nSP"),
        
]
        
    def __init__(self):

        self.shortenNamesCatalog={}
        return


    def build_qserv_shortenNames(self,catalogName,paramList):

        if catalogName in self.shortenNamesCatalog:
            del self.shortenNamesCatalog[catalogName]

        self.shortenNamesCatalog[catalogName]={}
        for p in paramList:
            p_init=p
            for i,v in enumerate(self.qserv_shortNameKeys_prefix):
                v1,v2=v
 #               v2="s%d_"%i
                if p.startswith(v1): p=p.replace(v1,v2)
            for i,v in enumerate(self.qserv_shortNameKeys_suffix):
                v1,v2=v
 #               v2="_p%d"%i
                if p.endswith(v1): p=p.replace(v1,v2)                
            self.shortenNamesCatalog[catalogName][p_init]=p
            print(p_init," ",p)
        self.check_shortenNames_integrity(catalogName)
        return

    def get_qserv_shortenNames(self,catalogName):

        return self.shortenNamesCatalog[catalogName]

    def get_reversed_qserv_shortenNames(self,catalogName):

        inv_names = {v: k for k, v in self.shortenNamesCatalog.items()}
        return inv_names

    def check_shortenNames_integrity(self,catalogName):

        nbError=0
        for pInitName in self.shortenNamesCatalog[catalogName]:
            p=self.shortenNamesCatalog[catalogName][pInitName]
            p_copy=self.shortenNamesCatalog[catalogName][pInitName]
            for i,v in enumerate(self.qserv_shortNameKeys_prefix):
                v1,v2=v
#                v2="s%d_"%i
                if p.startswith(v2): p=p.replace(v2,v1)
            for i,v in enumerate(self.qserv_shortNameKeys_suffix):
                v1,v2=v
 #               v2="_p%d"%i                
                if p.endswith(v2): p=p.replace(v2,v1)                

            if p!=pInitName:
                print("non reversible short name : ",p_copy," : ",p," -> ",pInitName)
                nbError+=1

        if nbError>0: sys.exit()



