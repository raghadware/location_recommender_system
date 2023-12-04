var map = L.map('map').setView([24.42, 43.58], 5);


L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

var markersLayer = L.layerGroup().addTo(map);
var recMarkersLayer = L.layerGroup().addTo(map);

// Add markers to the layer group
// var marker1 = L.marker([51.505, -0.09]).addTo(markersLayer);

function addMarker(e){
    // Add marker to map at click location; add popup window
    new L.marker(e.latlng).addTo(markersLayer);
    const url = `recommend/?lat=${encodeURIComponent(e.latlng.lat)}&lon=${encodeURIComponent(e.latlng.lng)}`;

    fetch(url)
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(jsonData => {
        // Handle the response data
        console.log(jsonData);
        for (var key in jsonData) {
            if (jsonData.hasOwnProperty(key)) {
                var entry = jsonData[key];
        

                var marker = L.marker([entry.lat, entry.lon]).addTo(recMarkersLayer);
                    // { icon: L.divIcon({ 
                    //     iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                    //     shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                    //     iconSize: [25, 41],
                    //     iconAnchor: [12, 41],
                    //     popupAnchor: [1, -34],
                    //     shadowSize: [41, 41]
                    // }) }
        
                // You can customize the marker popup content here if needed
                marker.bindPopup('<b>Info</b><br><b>Lat: </b>' 
                + entry.lat + '<b><br>Lon: </b>' + entry.lon
                +'<b><br>PVOUT_csi: </b>' + entry.PVOUT_csi
                +'<b><br>DNI: </b>' + entry.DNI
                +'<b><br>GHI: </b>' + entry.GHI
                +'<b><br>GTI_opta: </b>' + entry.GTI_opta
                +'<b><br>OPTA: </b>' + entry.OPTA
                +'<b><br>TEMP: </b>' + entry.TEMP
                +'<b><br>ELE: </b>' + entry.ELE
                );

                var style = document.createElement('style');
                style.innerHTML = '.blue-marker { background-color: blue; }';
                document.head.appendChild(style);
            }
        }
    })
    .catch(error => {
        // Handle errors
        console.error('Error:', error);
    });
};

map.addEventListener('click', (e)=>{
    markersLayer.clearLayers();
    recMarkersLayer.clearLayers();
    addMarker(e)})




