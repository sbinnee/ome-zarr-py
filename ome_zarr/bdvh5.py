from typing import Dict
from functools import cached_property
from pathlib import Path
import numpy as np
import h5py
import zarr

class BigDataViewerHDF5:
    """Intermediate class to convert BigDataViewer's HDF5 file to OMERO
    compatible .ome.zarr format.

    This class is just a wrapper to easily access BDV's h5 format and get
    information needed to convert it to .ome.zarr format.

    The group structure is probably designed to be compatible with BDV.

    """
    # Example
    # -------
    # *.h5
    # /
    #  ├── __DATA_TYPES__
    #  │   ├── Enum_Boolean
    #  │   └── String_VariableLength
    #  ├── s00
    #  │   ├── resolutions (6, 3) float64
    #  │   └── subdivisions (6, 3) int32
    #  ├── s01
    #  │   ├── resolutions (6, 3) float64
    #  │   └── subdivisions (6, 3) int32
    #  ├── s02
    #  │   ├── resolutions (6, 3) float64
    #  │   └── subdivisions (6, 3) int32
    #  └── t00000
    #      ├── s00
    #      │   ├── 0
    #      │   │   └── cells (1695, 4759, 7691) int16
    #      │   ├── 1
    #      │   │   └── cells (1695, 2379, 3845) int16
    #      │   ├── 2
    #      │   │   └── cells (847, 1189, 1922) int16
    #      │   ├── 3
    #      │   │   └── cells (423, 594, 961) int16
    #      │   ├── 4
    #      │   │   └── cells (211, 297, 480) int16
    #      │   └── 5
    #      │       └── cells (105, 148, 240) int16
    #      ├── s01
    #      │   ├── 0
    #      │   │   └── cells (1695, 4759, 7691) int16
    #      │   ├── 1
    #      │   │   └── cells (1695, 2379, 3845) int16
    #      │   ├── 2
    #      │   │   └── cells (847, 1189, 1922) int16
    #      │   ├── 3
    #      │   │   └── cells (423, 594, 961) int16
    #      │   ├── 4
    #      │   │   └── cells (211, 297, 480) int16
    #      │   └── 5
    #      │       └── cells (105, 148, 240) int16
    #      └── s02
    #          ├── 0
    #          │   └── cells (1695, 4759, 7691) int16
    #          ├── 1
    #          │   └── cells (1695, 2379, 3845) int16
    #          ├── 2
    #          │   └── cells (847, 1189, 1922) int16
    #          ├── 3
    #          │   └── cells (423, 594, 961) int16
    #          ├── 4
    #          │   └── cells (211, 297, 480) int16
    #          └── 5
    #              └── cells (105, 148, 240) int16

    def __init__(self, filename, mode=None):
        """
        Arguments
        ---------
        filename : str, path-like
            Path to .h5 file to read
        mode : str, {'r', 'r+', 'w', 'w-', 'x', 'a'}
            File mode

        """
        self.filename = filename
        self.h5 = h5py.File(filename)

    def __repr__(self):
        return "%s<%s, tseries_key=%s, channel_keys=%s, multiscales=%s>" % (
            self.__class__.__name__,
            self.filename,
            self.tseries_keys,
            self.channel_keys,
            self.multiscales,
        )

    def __getitem__(self, indices):
        """Read an array into memory given indices and resolution

        Checking `self.array` before getting an array is recommended.

        Arguments
        ---------
        indices : in order of (res,t,ch,z,y,x)
            `res`, `t` and `ch` should be a single integer and `z`, `y`, `x` are
            Slice. For `res`, 0 is the largest resolution.

        """
        res, t, ch, z, y, x = indices
        t = self.tseries_keys[t]
        ch = self.channel_keys[ch]

        arrays = self.arrays[t][ch][res][z, y, x]
        return arrays

    def to_zarr(self, filename, overwrite=False):
        if not isinstance(filename, Path):
            filename = Path(filename)
        # if not filename.exists():
        #     filename.mkdir()

        root = zarr.group(store=filename, overwrite=overwrite)

        # Create empty zarr array first
        for i in self.multiscales:
            # Assume all multiscales have the same size of t and ch
            dset = self.arrays[self.tseries_keys[0]][self.channel_keys[0]][i]
            new_shape = (
                len(self.tseries_keys), len(self.channel_keys), *dset.shape
            )
            new_chunks = (1, 1, *dset.chunks)
            _arr = root.empty_like(
                i,
                dset,
                shape=new_shape,
                chunks=new_chunks,
            )
            for _t in range(len(self.tseries_keys)):
                for _ch in range(len(self.channel_keys)):
                    _arr[_t, _ch, :] = dset

    @property
    def channel_keys(self):
        return tuple(
            filter(lambda s: s.startswith('s'), self.h5.keys())
        )

    @property
    def tseries_keys(self):
        return tuple(
            filter(lambda s: s.startswith('t'), self.h5.keys())
        )

    @property
    def num_channels(self):
        return len(self.channel_keys)

    @property
    def num_tseries(self):
        return len(self.tseries_keys)

    @cached_property
    def scales(self) -> Dict[str,np.ndarray]:
        return dict(
            (c, self.h5[c]['resolutions'][:]) for c in self.channel_keys
        )

    @cached_property
    def multiscales(self):
        lengths = [len(_v) for _v in self.scales.values()]
        if len(set(lengths)) == 1:
            return list(range(lengths[0]))
        return lengths

    @cached_property
    def chunk_sizes(self) -> Dict[str,np.ndarray]:
        return dict(
            (c, self.h5[c]['subdivisions'][:]) for c in self.channel_keys
        )

    @cached_property
    def arrays(self) -> Dict[str,Dict[str,h5py.Dataset]]:
        _arrays: Dict[str,Dict[str,h5py.Dataset]] = dict(
            (_ts, dict()) for _ts in self.tseries_keys
        )
        for _ts in self.tseries_keys:
            for _ch in self.channel_keys:
                _arrays[_ts][_ch] = [
                    self.h5[_ts][_ch][str(i)]['cells'] for i in range(
                        len(self.scales[_ch])
                    )
                ]
        return _arrays
