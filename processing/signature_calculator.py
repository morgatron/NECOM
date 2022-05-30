""" Monitor the stream of 'traces' ('frames'?) and calculate 'signatures' for various compoments.

Not yet decided the best way to do this. There are at least a few possibilities I can think of.

* Use some ind of initial guess at signatures, use those to fit out signals and update to better signatures, and iterate.

Pseudo-code:
signatures = np.array([square(), square(), square()])
for k in range(10):
    fit = sm.OLS(X, signatures).fit()
    signatures = np.sum(fit.params*X, axis=?)
    print(fit.residuals)

* Use Independant Component Analysis to obtain components automagically. Perhaps with some kind of refining step,
like subtracting off the parts that fit best using the above techniques then applying ICA to the remainder.

"""