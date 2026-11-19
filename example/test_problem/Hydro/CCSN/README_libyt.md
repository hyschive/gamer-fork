# Purpose

The provided `yt_inline.py` extracts slice data of various quantities on the XY, YZ, and XZ planes using the `save_as_dataset()` function in yt.

This workflow currently uses yt version 4.5.dev0.


# Troubleshooting
## Using `save_as_dataset()` in parallel mode in the inline script

To enable parallel data dumping, the installed yt package must be manually modified.

In `yt/frontends/ytdata/utilities.py`:

1. Ensure that both the `HDF5` library and the `h5py` Python package are installed with MPI supports.

2. Add the following import:

~~~ python
from mpi4py import MPI
~~~

3. Replace the file-opening line:

~~~ python
-   fh = h5py.File(filename, mode="w")
+   fh = h5py.File(filename, mode="w", driver="mpio", comm=MPI.COMM_WORLD)
~~~


## Failed when rendering the dumped `YT.Slice` objects

Data written via `save_as_dataset()` may fail during post-processing when the dataset contains a very large number of cells.

To resolve this issue, a patch is required in yt. Modify the `_read_particle_fields()` method of the `IOHandlerYTSpatialPlotHDF5` class in `yt/frontends/ytdata/io.py`. Replace the method with the following implementation:

~~~ python
def _read_particle_fields(self, chunks, ptf, selector):
    # Now we have all the sizes, and we can allocate
    for data_file in self._sorted_chunk_iterator(chunks):
        index_mask = slice(data_file.start, data_file.end)
        all_count = self._count_particles(data_file)
        with h5py.File(data_file.filename, mode="r") as f:
            for ptype, field_list in sorted(ptf.items()):
                if selector is None or getattr(selector, "is_all_data", False):
                    mask = index_mask
                else:
                    x = _get_position_array(ptype, f, "px")
                    y = _get_position_array(ptype, f, "py")
                    z = (
                        np.zeros(all_count[ptype], dtype="float64")
                        + self.ds.domain_left_edge[2].to("code_length").d
                    )
                    mask = selector.select_points(x, y, z, 0.0)
                    del x, y, z
                    if mask is None:
                        continue

                for field in field_list:
                    data = f[ptype][field][mask].astype("float64")
                    yield (ptype, field), data
~~~
