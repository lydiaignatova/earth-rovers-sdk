const express = require('express');
const app = express();
const port = 3000;
const cors = require('cors');
app.use(cors()); // This will allow all origins by default

// Variables to store the last received data
let lastMessage = 'No message received yet';
let lastData = 'Nothing';
let lat = '0.0';
let long = '0.0';
let ang = '0.0';

// Middleware to parse JSON data from requests
app.use(express.json());

// Middleware to parse URL-encoded data from form submissions
app.use(express.urlencoded({ extended: true }));

// Define a route for the root path ('/')
app.get('/', (req, res) => {
  res.send(`

    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GPS Server</title>
        <style>
            body {
                background-color: black;
                color: white; /* Make text white for better contrast */
                font-family: Arial, sans-serif;
            }
            button {
                background-color: #444; /* Dark background for buttons */
                color: white; /* White text */
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #666; /* Slightly lighter on hover */
            }
            input[type="text"] {
                padding: 10px;
                background-color: #333;
                color: white;
                border: none;
                border-radius: 5px;
                width: 100%;
            }
            .location-status {
                margin-top: 20px;
                font-size: 18px;
            }
            #compassHeadingIndicator {
                font-weight: bold;
            }
        </style>
    </head>
    <body>
    
    <h1>GPS Server</h1>
    
        <button id="reqCompassButton">Start compass</button>
        <p>Started: <span id="compassHeadingIndicator">NO</span></p>
    
    
        <p>Last received message: ${lastMessage}</p>
        <p>Latitude: ${lat}</p>
        <p>Longitude: ${long}</p>
        <p>Heading: ${ang}</p>
    
        <button id="startSendingButton">Send My Location</button>
        <button id="pauseSendingButton" disabled>Pause Sending Location</button>
    
        <input type="text" id="messageInput" placeholder="Enter your message here" />
    
        <div id="locationStatus" class="location-status">
            Location access status: <span id="locationAccessStatus">Checking...</span>
        </div>
    
        
        <script>
        // Variables to store location and interval ID
        let positionData = null;
        let compassData = "starter";
    
        let messageInterval = null;
        let isSendingLocation = false;
    
        const isIOS = /iPod|iPhone|iPad/.test(navigator.userAgent) && /AppleWebKit/.test(navigator.userAgent);
    
        // Button Logic
        function startCompass() {
            // Access to orientation 
            if (isIOS) {
                DeviceOrientationEvent.requestPermission()
                    .then((response) => {
                if (response === "granted") {
                    document.getElementById('compassHeadingIndicator').innerText = "granted orientation";
                    window.addEventListener("deviceorientation", compassHandler, true);
                } else {
                    document.getElementById('compassHeadingIndicator').innerText = "NOT granted orientation";
    
                    alert("has to be allowed!");
                }
                    })
                    .catch(() => alert("not supported"));
                } else {
                    document.getElementById('compassHeadingIndicator').innerText = "NOT an iphone orientation";
                    window.addEventListener("deviceorientationabsolute", compassHandler, true);
                }
    
                if (navigator.geolocation) {
                    // This will call the success callback every time the location changes
                    navigator.geolocation.watchPosition(function (position) {
                        positionData = position;
                    }, function (error) {
                        alert('Error getting location: ' + error.message);
                    }, {
                        enableHighAccuracy: true,  // Request high-accuracy data
                        timeout: 1000,             // Timeout if no position is available in 5 seconds
                        maximumAge: 0              // Don't use cached position data
                    });
                } else {
                    alert('Geolocation is not supported by this browser.');
                }
            }
    
        function sendMessages () {
            if (!isSendingLocation) {
                // Start sending location every 333 milliseconds (approximately 3 times per second)
                messageInterval = setInterval(sendLocation, 333); // 1000ms / 3 ≈ 333ms
                isSendingLocation = true;
    
                // Update button states
                document.getElementById("startSendingButton").disabled = true;
                document.getElementById("startSendingButton").innerText = "Sending Location...";
                document.getElementById("pauseSendingButton").disabled = false;
            }
        }
    
        function pauseMessages() {
            if (isSendingLocation) {
                clearInterval(messageInterval); // Stop sending location
                isSendingLocation = false;
    
                // Update button states
                document.getElementById("startSendingButton").disabled = false;
                document.getElementById("startSendingButton").innerText = "Send My Location";
                document.getElementById("pauseSendingButton").disabled = true;
            }
        }
    
        // Getting Location Data
        function compassHandler(e) {
            // Update the device tilt (gamma) on the webpage
            document.getElementById('compassHeadingIndicator').innerText = "compass accessed";
        
            let alpha;
            if (isIOS) {
                alpha = e.webkitCompassHeading; // For iOS
            } else {
                alpha = e.alpha; // For non-iOS (Android)
            }
            
            compassData = "teehee"
            // Update the compass heading (alpha) on the webpage
            if (alpha !== null && alpha != undefined) {
                const compass = alpha 
                compassData = compass;
                // compassData = "flowers";
            } else {
                compassData = "yikers"
            }
        }
    
        function checkLocationPermission() {
            if ("geolocation" in navigator) {
            navigator.geolocation.getCurrentPosition(
                () => {
                    // Location access granted
                    locationAccessStatus.textContent = "Granted";
                },
                () => {
                    // Location access denied
                    locationAccessStatus.textContent = "Denied";
                }
            );
            } else {
            locationAccessStatus.textContent = "Not available";
            }
        }
    
        // function getLocation() {
        //     if (navigator.geolocation) {
        //         navigator.geolocation.getCurrentPosition(function (position) {
        //         positionData = position;
        //         }, function (error) {
        //         alert('Error getting location: ' + error.message);
        //         });
        //     } else {
        //         alert('Geolocation is not supported by this browser.');
        //     }
        // }
    
    
    
        // Sending Messages
        function sendLocation() {
            // getLocation();
    
            if (positionData) {
                const userMessage = document.getElementById("messageInput").value;
    
                const data = {
                message: userMessage,
                latitude: positionData.coords.latitude,
                longitude: positionData.coords.longitude,
                orientation: compassData,
                };
    
                fetch('/message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(data => console.log('Location sent successfully:', data))
                .catch(error => console.error('Error sending location:', error));
            } else {
                console.log("Position data is not available yet.");
            }
        }
    
    
    
        // Putting it all together 
        function init() {
            document.getElementById('reqCompassButton').addEventListener("click", startCompass);
            document.getElementById('startSendingButton').addEventListener("click", sendMessages);
            document.getElementById('pauseSendingButton').addEventListener("click", pauseMessages);
            
            checkLocationPermission();
        }
    
        init(); 
    
        </script>
    </body>
    </html>

  `);
});

// Define a route to handle GET requests for the last message
app.get('/last-message', (req, res) => {
  res.send(lastMessage); // Return only the last received message
});

app.get('/last-data', (req, res) => {
  res.send(lastData); // Return only the last received data
});

// Route to handle POST requests to '/message'
app.post('/message', (req, res) => {
  lastMessage = req.body.message; // Store the received message
  lat = req.body.latitude;
  long = req.body.longitude;
  ang = req.body.orientation;
  lastData = req.body; // Store the entire data from the form

  console.log('Lat', lat);
  console.log('Lon', long);
  console.log('Ang', ang);
  console.log('Message', lastMessage);
  res.redirect('/'); // Redirect back to the homepage after receiving the message});
});

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
