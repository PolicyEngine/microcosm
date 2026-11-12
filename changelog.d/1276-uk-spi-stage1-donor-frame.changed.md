SPI stage 1 of `hmrc_spi_income_spine` now fits on a donor frame that holds every rare output's carriers (`sample_design: rare_carriers_certain`). An output is rare when its tape carriers number fewer than a quarter of the frame's rows. Every record that carries a rare output is in the frame. The other rows are drawn in proportion to FACT without replacement, each weighted at FACT over its inclusion probability, so the frame's weighted incidence is the tape's and no record repeats.

The stage's sign gates declare `cross_fitted_balanced` and `gate_level_columns`: SPI age band (the composite records' band among them), main source, has pay and has trade. Each categorical's one-hot columns go to the gate as one group, so its heaviest level is the reference. The FRS reference forests that place each copy's draw take the same gate setting. A copy's sign uniform is its twin's position in the reference gate, so the copies keep the tape gate's rate within a level only where the reference gate holds the FRS rows' rate there.

The stage evidence records two things:
- the frame (`donor_frame`): its counts of tape records rounded as every stage receipt's are, and each output's share of the tape's carriers the frame keeps;
- the gates' levels (`gate_levels`): the levels each gate offset, what each level fit came to, and the settings behind them.
