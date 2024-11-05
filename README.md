# DS.v2.5.3.3.5

```shell
curl --location 'https://spaceship-prediction-953e7e237ee4.herokuapp.com/predict' \
--header 'Content-Type: application/json' \
--data '{
    "features": [
        ["0029_01", "Europa", true, "B/2/P", "55 Cancri e", 21.0, false, 0.0, 0.0, 0.0, 0.0, 0.0, "Aldah Ainserfle"],
        ["0029_01", "Europa", true, "B/2/P", "55 Cancri e", 21.0, false, 0.0, 0.0, 0.0, 0.0, 0.0, "Aldah Ainserfle"]
    ],
    "threshold": 0.96
}
'
```

output:
`
{
  "prediction": [
    1,
    1
  ],
  "probability": [
    0.9741316826843636,
    0.9741316826843636
  ]
}
`