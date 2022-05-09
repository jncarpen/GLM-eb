# Code to add the expression for the GLM to the squish_statemat() method in glmmod.py.
# I have removed it for now to save some time/space. JCarpenter May 9, 2022.

def squish_statemat(self, spiketrain, stateIn, modelType='PE'):
        """ Combine state matrices for multivariate models and compose
        expression for calculating rate parameter. Spiketrain should be counts, and
        not smoothed.

        Parameters
        ----------
        spiketrain : np array
            speed-thresholded spiketrain (counts)
        stateIn : np array or list of np arrays
            for example [posgrid,ebgrid]
        stateIn : str
            model label, for example 'PE'

        Returns
        -------
        df
            dataframe with response variable and state matrix
        expr
            expression for the model of interest
        """
        if modelType == 'PE':
            posgrid = stateIn[0]; ebgrid = stateIn[1]
            ntime,nbins_eb = np.shape(ebgrid)
            _,nbins_p = np.shape(posgrid)
            A = np.zeros((ntime, nbins_p+nbins_eb)) #P+EB
            A[:,0:nbins_p] = posgrid; A[:,nbins_p:] = ebgrid
            df = pd.DataFrame(A)
            
            # name columns & get expression
            colnames = [];
            expr = 'y ~ '
            
            for i in range(nbins_p):
                val = str(i);
                expr = expr + 'P' + val + ' + '
                colnames.append('P' + val)
                
            for i in range(nbins_eb-1):
                val = str(i);
                expr = expr + 'E' + val + ' + '
                colnames.append('E' + val)
            expr = expr + 'E9'
            colnames.append('E9')
            df.columns = colnames
            
        elif modelType == 'P':
            ntime,nbins = np.shape(stateIn)
            df = pd.DataFrame(stateIn)
            colnames = [];
            expr = 'y ~ '
            
            for i in range(nbins-1):
                val = str(i);
                expr = expr + 'P' + val + ' + '
                colnames.append('P' + val)
            expr = expr + 'P99'
            colnames.append('P99')
            df.columns = colnames
            
        elif modelType == 'E':
            ntime,nbins = np.shape(stateIn)
            df = pd.DataFrame(stateIn)
            colnames = [];
            expr = 'y ~ '
            
            for i in range(nbins-1):
                val = str(i);
                expr = expr + 'E' + val + ' + '
                colnames.append('E' + val)
            expr = expr + 'E9'
            colnames.append('E9')
            df.columns = colnames

        elif modelType == 'H':
            ntime,nbins = np.shape(stateIn)
            df = pd.DataFrame(stateIn)
            colnames = [];
            expr = 'y ~ '
            
            for i in range(nbins-1):
                val = str(i);
                expr = expr + 'H' + val + ' + '
                colnames.append('H' + val)
            expr = expr + 'H9'
            colnames.append('H9')
            df.columns = colnames

        elif modelType == 'PH':
            posgrid = stateIn[0]; hdgrid = stateIn[1]
            ntime,nbins_hd = np.shape(hdgrid)
            _,nbins_p = np.shape(posgrid)
            A = np.zeros((ntime, nbins_p+nbins_hd)) #P+EB
            A[:,0:nbins_p] = posgrid; A[:,nbins_p:] = hdgrid
            df = pd.DataFrame(A)
            
            # name columns & get expression
            colnames = [];
            expr = 'y ~ '
            
            for i in range(nbins_p):
                val = str(i);
                expr = expr + 'P' + val + ' + '
                colnames.append('P' + val)
                
            for i in range(nbins_hd-1):
                val = str(i);
                expr = expr + 'H' + val + ' + '
                colnames.append('H' + val)
            expr = expr + 'H9'
            colnames.append('H9')
            df.columns = colnames
            
        else:
            print('Error: model type must be "P", "E", "PE", "PH", or "H"')
            
        # ** Note: ADD AS OPTION **
        # 20-80 test-train split (vs. k-fold)
        # note: make this an option
            # mask = np.random.rand(len(df)) < 0.8
            # df_train = df[mask]; df_test = df[~mask]
        
        # insert [raw] spiketrain into dataframe
        df.insert(0, 'y', spiketrain)
        
        return df,expr

