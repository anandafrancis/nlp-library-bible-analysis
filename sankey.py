def code_mapping(df, src, targ):
    """ 
    Purpose:
        Map labels in source and target columns to integers 
        
    Args:
        df  (pandas dataframe) : dataframe containing source and target columns
        src (str) : name of source column for sankey mapping
        targ (str) : name of target column for sankey mapping
        
    Return:
        df (pandas dataframe): """
    
    labels = list(df[src]) + list(df[targ])
    codes = list(range(len(labels)))
    lcmap = dict(zip(labels, codes))
    df = df.replace({src: lcmap, targ: lcmap})

    
    return df, labels


#%%
import plotly.graph_objects as go


def make_sankey(df, src, targ, vals=None, thrs=None, **kwargs):
    
    '''
    Purpose:
        Input specific parameters from a dataframe to customize and make Sankey
        
    Args:
        df (pandas dataframe) : 
        src (str) : name of source column for sankey mapping
        targ (str) : name of target column for sankey mapping
        val (str) : name of value column for sankey mapping
        thrs (int) : value that value column has to be greater than to create 
                     threshold for sankey visualization
    
    Return:
        None, plots Sankey diagram
        
    '''

    
    # make a Sankey diagram with aggregated dataframe
    if vals:
        
        # aggregate data by 2 columns
        grouped = df.groupby([src, targ]).size().reset_index(name=vals)
        
        # filter out rows whose artist count is below some threshold
        grouped = grouped[grouped[vals] > thrs]
        
        df, labels = code_mapping(grouped, src, targ)
        value = df[vals]
    
    # make a Sankey diagram with original dataframe
    else:
        
        df, labels = code_mapping(df, src, targ)
        value = [1] * df.shape[0]
    
    
    # plot and visualize  Sankey diagram
    link = {'source':df[src], 'target':df[targ], 'value':value}
    
    pad =kwargs.get('pad', 100)
    thickness = kwargs.get('thickness', 10)
    line_color = kwargs.get('line_color', 'black')
    width = kwargs.get('width', 2)

    node = {'pad':pad, 'thickness':thickness,
            'line':{'color':line_color, 'width':width},
            'label':labels}
    
    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)
    fig.show()
    
    