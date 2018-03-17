
# coding: utf-8

# In[1]:


import googlemaps

gmaps = googlemaps.Client(key="AIzaSyDXuGbsw9-qGKXsWGVQHIyqdDBnxJ1OhmA")


# In[21]:


def is_in_sg(lat, long):
    reverse_geocode_result = gmaps.reverse_geocode((lat, long))
    temp = list(filter(lambda x: 'country' in x['types'], reverse_geocode_result[0]['address_components']))
    return temp[0]['long_name'] == "Singapore"

