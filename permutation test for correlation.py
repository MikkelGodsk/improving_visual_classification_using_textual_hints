import numpy as np
import sys


def permutation_test(x,y, no_permutations=1000, verbose=True):
    """
        Performs a permutation test for correlation. Does so by permuting the
        y-values randomly in each trial, and then computing the correlation with
        x for the permuted y-values.
        
        Authored by Mikkel Godsk Jørgensen (s184399)
    """
    
    original_corr_coef = np.corrcoef(x, y)[0,1]
    if verbose:
        print("Correlation coefficient of data: {:f}".format(original_corr_coef))

    samples = x.size

    corr_coefs = np.zeros((no_permutations,))
    for i in range(no_permutations):
        # Compute correlation coefficient for permuted dataset
        corr_coefs[i] = np.corrcoef(x, 
                                    y[np.random.permutation(samples)])[0,1]
        if verbose:
            sys.stdout.flush()
            sys.stdout.write('\rProgress: {:.1f}%'.format((i+1)/no_permutations * 100))

    # Estimate p-value:  P(|r_permuted| > |r_data|)
    p_est = (np.sum(np.abs(corr_coefs) >= np.abs(original_corr_coef))+1) / (no_permutations+1)  # According to sklearn documentation
    
    sys.stdout.flush()
    sys.stdout.write('\r')
    if verbose:
        sys.stdout.write("\rEstimated p-value for the correlation: {:f}".format(p_est))
        print()
        
    return p_est


# Test functions of the permutation test
def test1():
    # Pure linear trend
    x = np.linspace(0,10,100)
    y = 3*x+5
    permutation_test(x, y)
    
def test2():
    # Noisy linear trend
    x = np.linspace(0,10,100)
    y = 3*x+5+np.random.normal(size=x.shape, scale=15)
    permutation_test(x,y)
    
def test3():
    # Pure noise
    x = np.linspace(0,10,100)
    y = np.random.normal(size=x.shape)
    permutation_test(x,y)
    
def test4():
    # Do we get a p-value of >=0.05 more than 5% of the time on average?
    counts = 0
    experiments = 2000   # experiments
    for i in range(experiments):
        sys.stdout.flush()
        sys.stdout.write('\rTest 4: Progress: {:.1f}%'.format((i+1)/experiments * 100))
        x = np.random.uniform(size=(100,))
        y = np.random.uniform(size=x.shape)
        p_est = permutation_test(x,y,no_permutations=100,verbose=False)
        if p_est <= 0.05:
            counts+=1
    sys.stdout.flush()
    sys.stdout.write('\r')
    print("Test 4: Got {:d} cases where p<=0.05 in {:d} trials ({:.2f}%)".format(counts, experiments, counts/experiments * 100))
    

if __name__=='__main__':
    # Unit tests just to see if the function behaves reasonably
    test1()
    test2()
    test3()
    for i in range(4):
        test4()
