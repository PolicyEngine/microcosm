Hash object-dtype graph storage by content instead of PyObject pointers, so a population column stays storage-equal to its own persisted-and-reloaded self.
