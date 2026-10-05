# Frozen legacy-lcoa source: commit cd56c11 (6 June 2023)

Repository: https://github.com/cpalazzi/lcoa-opt.git (local clone:
`/Users/carlopalazzi/programming/shipping_sprint/lcoa_model/lcoa-opt`).
Commit `cd56c11f10636e401928733b62f276a0e9f2c838`, tree
`a580a1ffcd39e835ab5ea3d1ad7b81af735731ca`, author nsalmon11,
"Commit before adding alternative grid supply cost for ammonia plant".
This is the last commit by the original author before Carlo Palazzi's
changes (first on 31 August 2023). Extracted with `git archive` on
15 September 2026; nothing edited.

`GeneralSteelData_20230831.xlsx` is not part of the commit. It is the cost
workbook first committed by Carlo on 31 August 2023 (5418ad1), byte-identical
to the working-tree file (md5 cee43cbf24814d35de7494a23cf1ed07). Its "Costs"
sheet holds annualised USD2018 costs at 8 % discount rate, 20 years, 2 % O&M
(see its "Discount Rate Calculation" sheet), on the RCP4.5 / Way "slow
transition" trajectory (its "Cost Trajectories" sheet matches
`x_SlowTransitionCosts.xlsx` and the RCP 4.5 sheet of `x_Cost Forecasting.xlsx`).
An intermediate committed version (a3d6533, 17 October 2023) has identical
Costs and Efficiencies sheets.
