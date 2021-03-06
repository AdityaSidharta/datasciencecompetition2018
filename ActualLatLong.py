import numpy as np
import math

def displace(lat, lng, theta, distance):

    """
    Displace a LatLng theta degrees counterclockwise and some
    meters in that direction.
    Notes:
        http://www.movable-type.co.uk/scripts/latlong.html
        0 DEGREES IS THE VERTICAL Y AXIS! IMPORTANT!
    Args:
        theta:    A number in degrees.
        distance: A number in meters.
    Returns:
        A new LatLng.
    """
    E_RADIUS = 6371000

    theta = np.float32(theta)

    delta = np.divide(np.float32(distance), np.float32(E_RADIUS))

    def to_radians(theta):
        return np.divide(np.dot(theta, np.pi), np.float32(180.0))

    def to_degrees(theta):
        return np.divide(np.dot(theta, np.float32(180.0)), np.pi)

    theta = to_radians(theta)
    lat1 = to_radians(lat)
    lng1 = to_radians(lng)
    c = np.sin(lat1) * np.cos(delta) + np.cos(lat1) * np.sin(delta) * np.cos(theta)
    lat2 = np.arcsin(c)

    lng2 = lng1 + np.arctan2( np.sin(theta) * np.sin(delta) * np.cos(lat1),
                              np.cos(delta) - np.sin(lat1) * np.sin(lat2))

    lng2 = (lng2 + 3 * np.pi) % (2 * np.pi) - np.pi

    return (np.round(to_degrees(lat2), decimals=4), np.round(to_degrees(lng2), decimals=4))

ORIGIN_LAT = 1.980
ORIGIN_LONG = 103.338
DISTANCE = 292

def find_distance_angle(lat, long):
    distance = math.sqrt(lat*lat + long*long) * 292
    angle_rad = math.atan2(long, lat)
    angle_deg = (angle_rad if angle_rad > 0 else (2*math.pi + angle_rad)) * 360 / (2*math.pi)
    return distance, angle_deg

new_lat_long = []

for y in range(0, -480, -1):
    for x in range(480):
        distance, angle = find_distance_angle(y, x)
        lat, long = displace(ORIGIN_LAT, ORIGIN_LONG, angle, distance)
        new_lat_long.append((lat, long))


