# Categorical probability artifacts

`microcosm.fit.categorical.fit_categorical` fits a real scikit-learn
`HistGradientBoostingClassifier` with directly supplied typed DESIGN weights.
It does not resample donors. Every declared class must have strictly positive
finite weighted support; missing or extra labels refuse. At least two classes
are required because the supported estimator's binary probability interface
does not provide a one-column single-class result. There is no invented mass
or fallback. The caller owns donor selection, scientific encoding and source
qualification.

Inputs use a unique ordered int64 entity index and finite float64 predictors.
The label Series must have the same index. `CategoricalConfig` exposes the
supported iteration, learning-rate, tree-size, regularization and bin controls.
All constructor parameters are serialized, including the seed, disabled early
stopping, absent class weighting and absent categorical encoding. Probabilities
are taken from the actual fitted estimator and reordered by its `classes_` into
the declared class order. Application neither fits recipients nor draws random
values.

The immutable fitted artifact binds ordered feature/class schemas, exact donor
matrix, labels, DESIGN weights, source projection hash, configuration, seed,
implementation and runtime identity. It contains the actual serialized fitted
estimator. Its loader is explicitly for trusted local producers/content stores:
the versioned envelope and expected digest detect corruption, but cannot make
an untrusted pickle safe. No fitted mutable object is returned to callers.

`graph_categorical.categorical_fit_node` consumes a donor Slice and typed source
projection. `categorical_probability_node` consumes its model/metadata pair,
the maintained typed recipient matrix, and a separate recipient source
projection. The nodes return artifacts only. Exact graph declarations,
producer-key pairing, platform, content hashes and training metadata are checked
before prediction. `read_probabilities` checks the caller's expected bindings,
recipient IDs/order, class order and probability values. Those descriptive
checks cannot grant survey-source authority.

This adapter changes no QRF behavior and attaches no Social Security amounts.
A future SS host must use the authenticated full-original source basis,
explicitly approved predictors and imported component order, then apply the
reason mask and conservation operator separately. The interpretation of
conditional category probabilities as shares, source-window transport,
beneficiary ownership and real-data statistical acceptance remain separate
development/release decisions.

The tiny invented tests exercise the real weighted estimator, typed graph
cold/required replay, class order, dependency invalidation and refusals. They
do not establish calibration, source admission, a full host replay or release
acceptance.
