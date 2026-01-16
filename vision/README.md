# Whiteboard CV

## Requirements

Install: 
 - [just](https://github.com/casey/just?tab=readme-ov-file#installation)
 - [uv](https://docs.astral.sh/uv/getting-started/installation/)

Clone this repo and open your terminal in the `vision` sub-directory.

## Usage

Run test on the included image: [test_img/3d-dev-kit.webp](test_img/3d-dev-kit.webp)

```
just test
```


Run test on your own image:

```
just detect ./path/to/some.jpg
```

See the results in the output: `debug_overlay.png` and `detections.json`
