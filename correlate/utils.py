def mass_wildcard_replace(string,*args):
        """
        Returns string with all values of *args replaced by '*'
        
        :param string: string to be replaced
        :param args: list of strings to return
        """
        for arg in args:
            string = string.replace(arg,"*")
        return string

def split_list(l,n):
    """
    Splits list l into n equal parts
    
    :param l: List to be split
    :param n: number of parts
    """
    for i in range(0, n):
        yield l[i::n]