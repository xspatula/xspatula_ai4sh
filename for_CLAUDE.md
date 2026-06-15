---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[3], line 8
      4 structured_process_D, scheme_params_D = Initiate_process(notebook_path, scheme_file, process_file)
      5 
      6 if structured_process_D is not None:
      7 
----> 8     Run_process(structured_process_D, scheme_params_D)

File ~/GitHub_xspatula/load_ai4sh_db/src/ai4sh/process.py:120, in Run_process(strcutured_process_D, scheme_params_D)
    118 if process_S.process.process == 'regression_modeling':
    119     model_C = Process_regression_model(process_S, pg_session_C)
--> 120     model_C._Sub_process(key)
    121 else:
    122     ml_C = Process_ml_preprocess(process_S, pg_session_C)

File ~/GitHub_xspatula/load_ai4sh_db/src/ai4sh/machine_learning_model.py:248, in Process_regression_model._Sub_process(self, _)
    246 def _Sub_process(self, _):
    247     if self.process_S.process.process == 'regression_modeling':
--> 248         self._Regression_modeling()

File ~/GitHub_xspatula/load_ai4sh_db/src/ai4sh/machine_learning_model.py:477, in Process_regression_model._Regression_modeling(self)
    475 imp = _feature_importance(model_tt, model_key)
    476 if imp is not None:
--> 477     fig2 = _plot_importance(imp, wavelengths, model_key, indicator, 'tt',
    478                             color=ind_color)
    479     _handle_fig(fig2,
    480                 os.path.join(plot_dir_tt,
    481                              'importance_%s_%s_tt.png' % (ind_s, model_key)),
    482                 show_feat, save_feat)
    484 pi = sk_permutation_importance(
    485     model_tt, X_te, y_te, n_repeats=10, random_state=42)

File ~/GitHub_xspatula/load_ai4sh_db/src/ai4sh/machine_learning_model.py:172, in _plot_importance(importances, wavelengths, model_key, indicator, label, color)
    170 bar_w = max(1, (wl_arr[1] - wl_arr[0]) * 0.8) if len(wl_arr) > 1 else 5
    171 fig, ax = plt.subplots(figsize=(9, 3))
--> 172 ax.bar(wl_arr, imp_arr, width=bar_w, color=color, alpha=0.75)
    173 ax.set_xlabel('Wavelength (nm)')
    174 ax.set_ylabel('Importance')

File /Applications/anaconda3/envs/xspatula_ai4sh_py_3.12/lib/python3.12/site-packages/matplotlib/__init__.py:1524, in _preprocess_data.<locals>.inner(ax, data, *args, **kwargs)
   1521 @functools.wraps(func)
   1522 def inner(ax, *args, data=None, **kwargs):
   1523     if data is None:
-> 1524         return func(
   1525             ax,
   1526             *map(cbook.sanitize_sequence, args),
   1527             **{k: cbook.sanitize_sequence(v) for k, v in kwargs.items()})
   1529     bound = new_sig.bind(ax, *args, **kwargs)
   1530     auto_label = (bound.arguments.get(label_namer)
   1531                   or bound.kwargs.get(label_namer))

File /Applications/anaconda3/envs/xspatula_ai4sh_py_3.12/lib/python3.12/site-packages/matplotlib/axes/_axes.py:2583, in Axes.bar(self, x, height, width, bottom, align, **kwargs)
   2580     if yerr is not None:
   2581         yerr = self._convert_dx(yerr, y0, y, self.convert_yunits)
-> 2583 x, height, width, y, linewidth, hatch = np.broadcast_arrays(
   2584     # Make args iterable too.
   2585     np.atleast_1d(x), height, width, y, linewidth, hatch)
   2587 # Now that units have been converted, set the tick locations.
   2588 if orientation == 'vertical':

File /Applications/anaconda3/envs/xspatula_ai4sh_py_3.12/lib/python3.12/site-packages/numpy/lib/_stride_tricks_impl.py:577, in broadcast_arrays(subok, *args)
    570 # nditer is not used here to avoid the limit of 64 arrays.
    571 # Otherwise, something like the following one-liner would suffice:
    572 # return np.nditer(args, flags=['multi_index', 'zerosize_ok'],
    573 #                  order='C').itviews
    575 args = [np.array(_m, copy=None, subok=subok) for _m in args]
--> 577 shape = _broadcast_shape(*args)
    579 result = [array if array.shape == shape
    580           else _broadcast_to(array, shape, subok=subok, readonly=False)
    581                           for array in args]
    582 return tuple(result)

File /Applications/anaconda3/envs/xspatula_ai4sh_py_3.12/lib/python3.12/site-packages/numpy/lib/_stride_tricks_impl.py:452, in _broadcast_shape(*args)
    447 """Returns the shape of the arrays that would result from broadcasting the
    448 supplied arrays against each other.
    449 """
    450 # use the old-iterator because np.nditer does not handle size 0 arrays
    451 # consistently
--> 452 b = np.broadcast(*args[:64])
    453 # unfortunately, it cannot handle 64 or more arguments directly
    454 for pos in range(64, len(args), 63):
    455     # ironically, np.broadcast does not properly handle np.broadcast
    456     # objects (it treats them as scalars)
    457     # use broadcasting to avoid allocating the full array

ValueError: shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (601,) and arg 1 with shape (601, 3).