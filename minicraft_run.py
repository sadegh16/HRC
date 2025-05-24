import pygame
import random

# Initialize pygame
pygame.init()
# Constants
WIDTH, HEIGHT = 800, 900
GRID_SIZE = 5
CELL_SIZE = WIDTH // (GRID_SIZE)
BACKPACK_SIZE = 4


# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
BROWN = (139, 69, 19)
BLACK = (0, 0, 0)

# Load images
agent_img = pygame.image.load('./assets/craftman.png')
stone_img = pygame.image.load('./assets/stone.png')
wood_img = pygame.image.load('./assets/wood.png')
pickaxe_img = pygame.image.load('./assets/pickaxe.png')
paper_img = pygame.image.load('./assets/paper.png')
paperplain_img = pygame.image.load('./assets/paperplain.png')
gold_img = pygame.image.load('./assets/goldore.png')

# Resize the images to fit the cell
agent_img = pygame.transform.scale(agent_img, (CELL_SIZE, CELL_SIZE))
stone_img = pygame.transform.scale(stone_img, (CELL_SIZE, CELL_SIZE))
wood_img = pygame.transform.scale(wood_img, (CELL_SIZE, CELL_SIZE))
pickaxe_img = pygame.transform.scale(pickaxe_img, (CELL_SIZE, CELL_SIZE))
paper_img = pygame.transform.scale(paper_img, (CELL_SIZE, CELL_SIZE))
paperplain_img = pygame.transform.scale(paperplain_img, (CELL_SIZE, CELL_SIZE))
gold_img = pygame.transform.scale(gold_img, (CELL_SIZE, CELL_SIZE))


FONT = pygame.font.Font(None, 36)


# Game classes
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.backpack = []

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def craft(self, location):
        if location == 'workshop' and 'stone' in self.backpack and 'wood' in self.backpack:
            self.backpack.append('pickaxe')
        elif location == 'mine' and 'pickaxe' in self.backpack:
            self.backpack.append('goldore')

    def pick(self, item):
        if item in ['stone', 'wood'] and len(self.backpack) < BACKPACK_SIZE:
            self.backpack.append(item)

    def draw(self, screen):
        FONT = pygame.font.Font(None, 36)

        screen.blit(agent_img, (self.x * CELL_SIZE, self.y * CELL_SIZE))
        # backpack_x = 10
        # for item in self.backpack:
        #     if item == 'stone':
        #         screen.blit(stone_img, (backpack_x, HEIGHT - CELL_SIZE))
        #     elif item == 'wood':
        #         screen.blit(wood_img, (backpack_x, HEIGHT - CELL_SIZE))
        #     elif item == 'pickaxe':
        #         screen.blit(pickaxe_img, (backpack_x, HEIGHT - CELL_SIZE))
        #     elif item == 'goldore':
        #         screen.blit(goldore_img, (backpack_x, HEIGHT - CELL_SIZE))
        #     backpack_x += CELL_SIZE

        # Draw backpack area
        pygame.draw.rect(screen, BLACK, (10, HEIGHT - 70, WIDTH - 20, 60))
        text = FONT.render('Backpack: ' + ', '.join(self.backpack), True, WHITE)
        screen.blit(text, (20, HEIGHT - 65))

# Game loop
def game_loop():
    global action_count
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Gridworld Game')
    clock = pygame.time.Clock()

    agent = Agent(GRID_SIZE // 2, GRID_SIZE // 2)

    workshop =  GRID_SIZE - 2, GRID_SIZE - 4
    mine = GRID_SIZE-3, GRID_SIZE - 2

    # Single random wood and stone location
    stone_pos = GRID_SIZE-4, GRID_SIZE - 3
    wood_pos = GRID_SIZE-1, GRID_SIZE - 3
    paper_pos = GRID_SIZE-5, GRID_SIZE - 1
    paperplain_pos = GRID_SIZE-1, GRID_SIZE - 5
    gold_pos = GRID_SIZE-3, GRID_SIZE - 1

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action_count += 1


                if event.key == pygame.K_UP and agent.y > 0:
                    agent.move(0, -1)
                if event.key == pygame.K_DOWN and agent.y < GRID_SIZE - 1:
                    agent.move(0, 1)
                if event.key == pygame.K_LEFT and agent.x > 0:
                    agent.move(-1, 0)
                if event.key == pygame.K_RIGHT and agent.x < GRID_SIZE - 1:
                    agent.move(1, 0)
                if event.key == pygame.K_c:
                    if (agent.x, agent.y) == workshop:
                        agent.craft('workshop')
                    elif (agent.x, agent.y) == mine:

                        agent.craft('mine')
                if event.key == pygame.K_p:
                    if (agent.x, agent.y) == stone_pos:
                        agent.pick('stone')
                        stone_pos = (-1, -1)  # Remove stone from grid
                    elif (agent.x, agent.y) == wood_pos:
                        agent.pick('wood')
                        wood_pos = (-1, -1)  # Remove wood from grid
                # Check if goldore is achieved
                if 'goldore' in agent.backpack:
                    running = False
                    end_message = "Congratulations! You achieved goldore!"

                # Check if action count exceeds limit
                elif action_count > 50:
                    running = False
                    end_message = "Game over! You took too many actions."

        screen.fill(WHITE)
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(screen, BLUE, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(screen, BLUE, (0, y), (WIDTH, y))

        # Drawing stone and wood:
        screen.blit(stone_img, (stone_pos[0] * CELL_SIZE, stone_pos[1] * CELL_SIZE))
        screen.blit(wood_img, (wood_pos[0] * CELL_SIZE, wood_pos[1] * CELL_SIZE))
        screen.blit(pickaxe_img, (mine[0] * CELL_SIZE, mine[1] * CELL_SIZE))
        screen.blit(paper_img, (paper_pos[0] * CELL_SIZE, paper_pos[1] * CELL_SIZE))
        screen.blit(paperplain_img, (paperplain_pos[0] * CELL_SIZE, paperplain_pos[1] * CELL_SIZE))
        screen.blit(gold_img, (gold_pos[0] * CELL_SIZE, gold_pos[1] * CELL_SIZE))

        agent.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    # Display end message
    screen.fill(WHITE)
    text = FONT.render(end_message, True, BLACK)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    pygame.time.wait(3000)  # Wait for 3 seconds to read the message

    # pygame.quit()

if __name__ == "__main__":
    while True:
        action_count = 0
        game_loop()
