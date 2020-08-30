import pygame
import sys
import time
from CSP_AI import CSP_AI as AI_CSP
from SAT_AI import SAT_AI as AI_SAT
from MineSweeperGame import Minesweeper
from configuration import WIDTH, HEIGHT, MINES, BLACK, AZURE, WHITE, size, height, width



pygame.init()
pygame.display.set_caption("SMART SWEEPER")
screen = pygame.display.set_mode(size)

OPEN_SANS = "graphics/fonts/OpenSans-Regular.ttf"
small_font = pygame.font.Font(OPEN_SANS, 20)
medium_font = pygame.font.Font(OPEN_SANS, 28)
large_font = pygame.font.Font(OPEN_SANS, 40)

game = Minesweeper(height=HEIGHT, width=WIDTH, mines=MINES)

mine_pressed = None

flags = set()
opened = set()
ai = None

CSP_AI = False
lost = False

show_instructions = True
first_move = True
make_ai_move = False
auto_play = False
autoplay_speed = 0.3

while True:

    for act in pygame.event.get():
        if act.type == pygame.QUIT:
            sys.exit()

    screen.fill(BLACK)

    if show_instructions:

        title = large_font.render("MineSweeper AI", True, WHITE)
        title_rect = title.get_rect()
        title_rect.center = ((width / 2), 50)
        screen.blit(title, title_rect)


        beginner_rect = pygame.Rect((width / 4), (1/5) * height, width / 2, 50)
        button_text = medium_font.render("Beginner", True, BLACK)
        button_text_rect = button_text.get_rect()
        button_text_rect.center = beginner_rect.center
        pygame.draw.rect(screen, WHITE, beginner_rect)
        screen.blit(button_text, button_text_rect)

        intermediate_rect = pygame.Rect((width / 4), (2/5) * height, width / 2, 50)
        button_text = medium_font.render("Intermediate", True, BLACK)
        button_text_rect = button_text.get_rect()
        button_text_rect.center = intermediate_rect.center
        pygame.draw.rect(screen, WHITE, intermediate_rect)
        screen.blit(button_text, button_text_rect)

        expert_rect = pygame.Rect((width / 4), (3 / 5) * height, width / 2, 50)
        button_text = medium_font.render("Expert", True, BLACK)
        button_text_rect = button_text.get_rect()
        button_text_rect.center = expert_rect.center
        pygame.draw.rect(screen, WHITE, expert_rect)
        screen.blit(button_text, button_text_rect)








        left_click, _, _ = pygame.mouse.get_pressed()
        if left_click == 1:
            mouse = pygame.mouse.get_pos()
            if beginner_rect.collidepoint(mouse):
                HEIGHT=8
                WIDTH=8
                MINES=10
                game = Minesweeper(height=8, width=8, mines=16)
                csp_ai = AI_CSP(HEIGHT, WIDTH, MINES)
                sat_ai = AI_SAT(HEIGHT, WIDTH, MINES)
                show_instructions = False

            elif intermediate_rect.collidepoint(mouse):
                HEIGHT = 16
                WIDTH = 16
                MINES = 40
                game = Minesweeper(height=16, width=16, mines=40)
                csp_ai = AI_CSP(HEIGHT, WIDTH, MINES)
                sat_ai = AI_SAT(HEIGHT, WIDTH, MINES)
                show_instructions = False

            elif expert_rect.collidepoint(mouse):
                HEIGHT = 20
                WIDTH = 20
                MINES = 99
                game = Minesweeper(height=20, width=20, mines=99)
                csp_ai = AI_CSP(HEIGHT, WIDTH, MINES)
                sat_ai = AI_SAT(HEIGHT, WIDTH, MINES)
                show_instructions = False
        pygame.display.flip()
        continue
    BOARD_PADDING = 20
    board_width = (  3/5*width) - (BOARD_PADDING * 2)
    board_height = height - (BOARD_PADDING * 2)
    cell_size = int(min(board_width / WIDTH, board_height / HEIGHT))
    board_origin = (BOARD_PADDING, BOARD_PADDING)
    flag = pygame.image.load("graphics/images/flag.png")
    flag = pygame.transform.scale(flag, (cell_size, cell_size))
    mine = pygame.image.load("graphics/images/mine.png")
    mine = pygame.transform.scale(mine, (cell_size, cell_size))
    mine_red = pygame.image.load("graphics/images/mine-red.png")
    mine_red = pygame.transform.scale(mine_red, (cell_size, cell_size))

    tiles = []
    for i in range(HEIGHT):
        row = []
        for j in range(WIDTH):

            # Draw rectangle for cell
            rect = pygame.Rect(
                board_origin[0] + j * cell_size,
                board_origin[1] + i * cell_size,
                cell_size, cell_size
            )
            pygame.draw.rect(screen, AZURE, rect)
            pygame.draw.rect(screen, WHITE, rect, 3)

            # Add a mine, flag, or number if needed
            if game.is_mine((i, j)) and lost:
                if (i, j) == mine_pressed:
                    screen.blit(mine_red, rect)
                else:
                    screen.blit(mine, rect)
            elif (i, j) in flags:
                screen.blit(flag, rect)
            elif (i, j) in opened:
                neighbors = small_font.render(
                    str(game.nearby_mines((i, j))),
                    True, BLACK
                )
                neighborsTextRect = neighbors.get_rect()
                neighborsTextRect.center = rect.center
                screen.blit(neighbors, neighborsTextRect)
            row.append(rect)
        tiles.append(row)

    sat_ai_btn = pygame.Rect(
        (2 / 3) * width + BOARD_PADDING, BOARD_PADDING,
        (width / 3) - BOARD_PADDING * 2, 50
    )
    b_text = "autoplay: SAT ai" if not auto_play else "PAUSE"
    button_text = medium_font.render(b_text, True, BLACK)
    beginner_rect = button_text.get_rect()
    beginner_rect.center = sat_ai_btn.center
    pygame.draw.rect(screen, WHITE, sat_ai_btn)
    screen.blit(button_text, beginner_rect)

    csp_ai_btn = pygame.Rect(
        (2 / 3) * width + BOARD_PADDING, BOARD_PADDING + 70,
        (width / 3) - BOARD_PADDING * 2, 50
    )
    b_text = "autoplay: CSP ai"
    CSP_buttonText = medium_font.render(b_text, True, BLACK)
    CSP_buttonRect = button_text.get_rect()
    CSP_buttonRect.center = csp_ai_btn.center
    if not auto_play:
        pygame.draw.rect(screen, WHITE, csp_ai_btn)
        screen.blit(CSP_buttonText, CSP_buttonRect)

    ai_button = pygame.Rect(
        (2 / 3) * width + BOARD_PADDING, BOARD_PADDING + 140,
        (width / 3) - BOARD_PADDING * 2, 50
    )
    button_text = medium_font.render("AI Move", True, BLACK)
    beginner_rect = button_text.get_rect()
    beginner_rect.center = ai_button.center
    if not auto_play:
        pygame.draw.rect(screen, WHITE, ai_button)
        screen.blit(button_text, beginner_rect)

    reset_button = pygame.Rect(
        (2 / 3) * width + BOARD_PADDING, BOARD_PADDING + 210,
        (width / 3) - BOARD_PADDING * 2, 50
    )
    button_text = medium_font.render("Reset", True, BLACK)
    beginner_rect = button_text.get_rect()
    beginner_rect.center = reset_button.center
    if not auto_play:
        pygame.draw.rect(screen, WHITE, reset_button)
        screen.blit(button_text, beginner_rect)

    text = "Lost" if lost else "Won" if game.mines == flags else ""
    text = medium_font.render(text, True, WHITE)
    text_rect = text.get_rect()
    text_rect.center = ((5 / 6) * width, BOARD_PADDING + 360)
    screen.blit(text, text_rect)

    action = None
    text = "number of mines: " + str(MINES - len(flags))
    text = small_font.render(text, True, WHITE)
    text_rect = text.get_rect()
    text_rect.center = ((5 / 6) * width, BOARD_PADDING + 460)
    screen.blit(text, text_rect)

    left, _, right = pygame.mouse.get_pressed()

    if right == 1 and not lost and not auto_play:
        mouse = pygame.mouse.get_pos()
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if tiles[i][j].collidepoint(mouse) and (i, j) not in opened:
                    if (i, j) in flags:
                        flags.remove((i, j))
                    else:
                        flags.add((i, j))

    elif left == 1:
        mouse = pygame.mouse.get_pos()

        if sat_ai_btn.collidepoint(mouse):
            ai = sat_ai

            if not lost:
                auto_play = not auto_play
            else:
                auto_play = False
            time.sleep(0.2)
            continue
        if csp_ai_btn.collidepoint(mouse):
            ai = csp_ai
            if not lost:
                auto_play = not CSP_AI

            else:
                auto_play = False
            continue

        elif ai_button.collidepoint(mouse) and not lost:
            make_ai_move = True
            ai = sat_ai

        elif reset_button.collidepoint(mouse):
            game = Minesweeper(height=HEIGHT, width=WIDTH, mines=MINES)
            ai = None
            sat_ai = AI_SAT(HEIGHT, WIDTH, MINES)
            csp_ai = AI_CSP(HEIGHT, WIDTH, MINES)
            first_move = True
            opened = set()
            flags = set()
            lost = False
            mine_pressed = None
            show_instructions=True
            continue

        elif not lost:
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    if (tiles[i][j].collidepoint(mouse)
                            and (i, j) not in flags
                            and (i, j) not in opened):
                        action = (i, j)
    if auto_play or make_ai_move:
        if make_ai_move:
            make_ai_move = False
        action = ai.make_safe_move()
        flags = ai.mines.copy()
        if action is None:
            action = ai.make_random_move()
            flags = ai.mines.copy()
            if action is None:
                print("No moves left to make.")
                auto_play = False
                if not lost:
                    print("WON")
            else:
                print("No known safe moves, AI making random move:", action)
        else:
            print("AI making safe move:", action)

    if action:
        if first_move:
            game = Minesweeper(height=HEIGHT, width=WIDTH, mines=MINES, first_move=action)
            first_move = False
        if not game.is_mine(action):
            nearby = game.nearby_mines(action)
            opened.add(action)
            if ai is None:
                csp_ai.extend_knowledge_base(action, nearby)
                sat_ai.extend_knowledge_base(action, nearby)
            else:
                ai.extend_knowledge_base(action, nearby)
        else:
            lost = True
            mine_pressed = action
            auto_play = False

    pygame.display.flip()