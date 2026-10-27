# Send to TurnPilot: an iOS Shortcut for Apple Maps

This shortcut adds TurnPilot to the share sheet in Apple Maps. Find a place in Maps, tap Share, tap Send to TurnPilot, and the comma starts navigating there. It posts to the same endpoint the phone app uses, so nothing else needs to run on the comma.

The steps follow the Shortcuts app on iOS 18. Nobody has tried them on a phone yet, so a menu name may differ on your version. The only part that has to match is the request described in [What the shortcut sends](#what-the-shortcut-sends), and the [test with curl](#test-the-endpoint-from-a-computer) checks the comma side without a phone.

## Before you start

1. Join the comma to your iPhone's Personal Hotspot, and keep the phone on that hotspot while you set this up.
2. Scan the QR code on the comma screen with the phone camera and open the link in Safari. The page shows the TurnPilot status panel. The link ends in `?t=` and a token, which is the password for the comma. Anyone on the hotspot who has the token can set the destination, so don't share it.
3. Optional: to add the app to the home screen, tap Share in Safari, then Add to Home Screen, while the address still ends in `?t=...`. The home-screen app doesn't share cookies with Safari, so the saved address is what pairs it. If you opened the page without the token, scan the QR code again.
4. On the phone app, open Shortcut setup at the bottom of the page. It shows the URL and the token for this comma. Keep the page open, because you copy both values below.

## Build the shortcut

1. Open the Shortcuts app, tap the plus button, and name the shortcut `Send to TurnPilot`.
2. Tap the shortcut name at the top, then Details (the info button). Turn on Show in Share Sheet. Under Receives, choose URLs and Text, plus Maps if your version lists it. Apple Maps shares a link to `maps.apple.com`, which arrives as a URL.
3. Add the action Get Contents of URL.
4. Set the URL field to the URL from Shortcut setup. It looks like `http://172.20.10.2:8088/api/destination`.
5. Tap Show More on the action, then set these fields:
   - Method: POST
   - Headers: add one header. The key is `Authorization` and the text is `Bearer ` followed by the token, with one space after Bearer.
   - Request Body: JSON
6. In the JSON body, tap Add new field and choose Text. Set the key to `url`. For the value, tap the field, choose Select Variable, and pick Shortcut Input.
7. Optional: add Get Dictionary Value after it, set to get the value for `name` in Contents of URL, then add Show Notification with the text `Navigating to ` followed by that value. Without these two actions the shortcut runs silently, and you confirm on the phone app or the comma screen.
8. Tap Done.

## Use it

1. Open Apple Maps and find a destination, either by search or by dropping a pin.
2. Tap the place, tap Share, and tap Send to TurnPilot. It may be under More if you haven't used it before. Tap Edit Actions on the share sheet to pin it near the top.
3. Open the TurnPilot page. The destination shows under To, and the route appears within a few seconds once the comma has a GPS fix and internet through the hotspot.

To clear the destination, tap Stop navigation on the page.

## What the shortcut sends

The shortcut makes this request. Any client that sends it works.

```
POST http://<comma address>:8088/api/destination
Authorization: Bearer <token>
Content-Type: application/json

{"url": "https://maps.apple.com/?ll=39.7684%2C-86.1581&q=Monument%20Circle"}
```

The server reads coordinates from the link in this order: `coordinate`, `ll`, `daddr`, `q`, then `sll`. If the link has none of them, it geocodes the address or name from the link with the same search the page uses. If the link is a short `maps.apple.com/p/...` address, the server follows it once and looks for coordinates in the result.

The reply is JSON. On success it is `{"ok": true, "name": "...", "latitude": ..., "longitude": ...}` and the destination is set. An error carries a message, for example `{"error": "no place found for 'xyz'"}` with status 404.

The endpoint also takes coordinates directly, which is the fallback if a link doesn't work:

```
{"latitude": 39.7684, "longitude": -86.1581, "name": "Monument Circle"}
```

To build the shortcut that way, replace the Get Contents of URL body with a Get Details of Locations action on the Shortcut Input (Latitude, Longitude and Name) and put those three values in the JSON fields `latitude`, `longitude` and `name`.

## Test the endpoint from a computer

On a computer on the same network as the comma, run this with your values. A successful call prints the JSON reply, and the destination appears on the page.

```
curl -X POST http://172.20.10.2:8088/api/destination \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://maps.apple.com/?ll=39.7684%2C-86.1581&q=Monument%20Circle"}'
```

## Troubleshooting

- The shortcut says it can't connect. The phone and the comma must be on the same hotspot. The comma's address can change after it reconnects, so copy the URL from Shortcut setup again and update the action.
- The reply is 401 unauthorized. The header must read `Bearer ` followed by the token with nothing else. Copy the token again from Shortcut setup.
- The reply is 415. The request body must be JSON. In the Get Contents of URL action, set Request Body to JSON rather than Form or File.
- The reply is 404 or 400. The link had no coordinates and the search found no match, or the text wasn't a maps link. Check the reply text, which says which.
- The destination shows on the page but no route appears. The route comes from navd, which needs a GPS fix and a route service reachable through the hotspot.
- To replace a leaked token, over SSH run `python3 -m openpilot.common.params TurnPilotWebToken <new value>` with a long random string. The server reads the token on every request, so the old one stops working at once, the QR code on the comma screen changes, and every phone has to pair again.
