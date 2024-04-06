import random
from math import pi

import numpy as np
import seaborn as sns
from PIL import Image, ImageColor, ImageDraw


def create_gradient(width, height, gradient, angle):
    # Create a numpy array to store the gradient values
    gradient_image = np.zeros((height, width, 4), dtype=np.uint8)

    # Loop through each row and column of the gradient array
    for y in range(height):
        for x in range(width):
            # Calculate the normalized position along the gradient line
            t = (x * np.cos(angle) + y * np.sin(angle)) / (
                width * np.cos(angle) + height * np.sin(angle)
            )
            # Clamp the position to the range [0, 1]
            t = max(0, min(1, t))
            # Find the two closest color stops
            i = 0
            while i < len(gradient) - 1 and t > gradient[i + 1][0]:
                i += 1
            # Interpolate the color and alpha values between the two stops
            c1 = gradient[i][1]
            c2 = gradient[i + 1][1]
            if isinstance(c1, str):
                c1 = ImageColor.getcolor(c1, "RGBA")
            if isinstance(c2, str):
                c2 = ImageColor.getcolor(c2, "RGBA")
            s1 = gradient[i][0]
            s2 = gradient[i + 1][0]
            f = (t - s1) / (s2 - s1)
            r = int(c1[0] + f * (c2[0] - c1[0]))
            g = int(c1[1] + f * (c2[1] - c1[1]))
            b = int(c1[2] + f * (c2[2] - c1[2]))
            a = int(c1[3] + f * (c2[3] - c1[3]))
            # Assign the color and alpha values to the gradient array
            gradient_image[y, x] = [r, g, b, a]

    # Convert the gradient array to an image
    gradient_image = Image.fromarray(gradient_image)

    return gradient_image


def create_logo_image(grid, filled_color, column_gradients, cell_size):
    image_width = len(grid[0]) * cell_size[0]
    image_height = len(grid) * cell_size[1]

    # Create a new image in RGBA mode
    img = Image.new("RGBA", (image_width, image_height))
    draw = ImageDraw.Draw(img)

    # Create a gradient for each column
    if column_gradients is not None:
        for i in range(len(grid[0])):
            gradient_image = create_gradient(
                cell_size[0],
                image_height,
                column_gradients[i % len(column_gradients)],
                pi / 2,
            )
            img.paste(gradient_image, (i * cell_size[0], 0), gradient_image)

    for i, row in enumerate(grid):
        # Define the starting and ending y coordinates for the row
        start_y = i * cell_size[1]
        end_y = start_y + cell_size[1]

        for j, cell in enumerate(row):
            # If cell is empty, skip it
            if cell == "X":
                # Define the starting and ending x coordinates for the cell
                start_x = j * cell_size[0]
                end_x = start_x + cell_size[0]
                draw.rectangle([(start_x, start_y), (end_x, end_y)], fill=filled_color)

    return img


def calculate_columns_gradients(
    grid, left_gradient, right_gradient, seed, add_alpha_gradient=True
):
    random.seed(seed)

    def add_stops(gradient):
        return [(i / (len(gradient) - 1), v) for i, v in enumerate(gradient)]

    def clip(val, min_val, max_val):
        return min(max(val, min_val), max_val)

    def color_mod_val():
        return random.random() * 16 - 8

    columns = len(grid[0])

    # Create interpolated gradients for each column
    columns_gradients = []
    left_gradient = add_stops(left_gradient)
    right_gradient = add_stops(right_gradient)
    for i in range(columns):
        gradient = []
        for j in range(len(left_gradient)):
            (r1, g1, b1) = left_gradient[j][1]
            (r2, g2, b2) = right_gradient[j][1]
            s = left_gradient[j][0] + (right_gradient[j][0] - left_gradient[j][0])
            r = int((r1 + (r2 - r1) * i / (columns - 1)) * 255 + color_mod_val())
            g = int((g1 + (g2 - g1) * i / (columns - 1)) * 255 + color_mod_val())
            b = int((b1 + (b2 - b1) * i / (columns - 1)) * 255 + color_mod_val())

            # Clip the color values to the range [0, 255]
            s = clip(s, 0, 1)
            r = clip(r, 0, 255)
            g = clip(g, 0, 255)
            b = clip(b, 0, 255)

            # if j > 0:
            #     r = min(r, gradient[-1][1][0])
            #     g = min(g, gradient[-1][1][1])
            #     b = min(b, gradient[-1][1][2])

            a = 255
            gradient.append((s, (r, g, b, a)))

        columns_gradients.append(gradient)

    return columns_gradients


if __name__ == "__main__":
    # Initial logo version
    grid = """
....................................
.....XXX.XXX.X...X.X.X...X.X..X.....
.....X...X.X.X...X.X.XX.XX.X..X.....
.X.X.X...X.X.X...X.X.X.X.X.XX.X..XX.
.X.X.X...X.X.X...X.X.X...X.X.XX.X...
..X..X...X.X.X...X.X.X...X.X..X..X..
.X.X.X...X.X.X...X.X.X...X.X..X...X.
.X.X.XXX.XXX.XXX.XXX.X...X.X..X.XX..
....................................
""".strip().split(
        "\n"
    )

    # Logo with the same number of filled cells in each row (k=13)
    logo_grid = """
....................................
.....XXX.XXX.X.......X...X.X..X..XX.
.....X...X.X.X.....X.XX.XX.XX.X.X...
.....X...X.X.X...X.X.X.X.X.X.XX..X..
.X.X.X...X.X.X...X.X.X...X.X..X...X.
..X..X...X.X.X...X.X.X...X.X..X.XX..
.X.X.X...XXX.X...X.X.X...X.X..X.....
.X.X.XXX.....XXX.XXX.X........X.....
....................................
""".strip().split(
        "\n"
    )

    favicon_grid = """
.........
.....XXX.
.....X...
.....X...
.X.X.X...
..X..X...
.X.X.X...
.X.X.XXX.
.........
""".strip().split(
        "\n"
    )

    # Count the number of filled cells in each row
    # for i, row in enumerate(grid):
    #     print(f"Full cells in row {i}: {row.count('X')}")

    cell_size = (20, 20)
    filled_color = (255, 255, 255, 255)  # RGB color for filled cells
    columns_gradients = calculate_columns_gradients(
        grid, sns.color_palette("crest"), sns.color_palette("flare"), 1993
    )

    # Generate the gradient image
    logo_image = create_logo_image(
        logo_grid, filled_color, columns_gradients, cell_size
    )
    logo_image.save("xCOLUMNs_logo.png")  # Save the image as 'xCOLUMNs_logo.png'

    # Generate the gradient image
    logo_image = create_logo_image(logo_grid, filled_color, None, cell_size)
    logo_image.save(
        "xCOLUMNs_logo_nobg.png"
    )  # Save the image as 'xCOLUMNs_logo_nobg.png'

    # Generate the favicon image
    favicon_image = create_logo_image(
        favicon_grid, filled_color, columns_gradients, cell_size
    )
    favicon_image.save("favicon.png")  # Save the image as 'favicon.png'
