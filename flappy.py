import pygame
import neat
import random
import os
import time
import pickle

FRAME_RATE=20

STD_VEL=5

pygame.font.init()
BIRD_X=230
BIRD_Y=350
BASE_Y=730

WIN_WIDTH=500
WIN_HEIGHT=800

FACTOR=2

STAT_FONT=pygame.font.SysFont("comicsans",50)
BIRD_IMGS=[pygame.transform.scale_by(pygame.image.load(r"E:\pycharm_all_projects\flappy_bird_neat\imgs_b286d95d6d\imgs\bird1.png"),FACTOR),pygame.transform.scale_by(pygame.image.load(r"E:\pycharm_all_projects\flappy_bird_neat\imgs_b286d95d6d\imgs\bird1.png"),FACTOR),pygame.transform.scale_by(pygame.image.load(r"E:\pycharm_all_projects\flappy_bird_neat\imgs_b286d95d6d\imgs\bird2.png"),FACTOR)]
PIPE_IMG=pygame.transform.scale_by(pygame.image.load(r"E:\pycharm_all_projects\flappy_bird_neat\imgs_b286d95d6d\imgs\pipe.png"),FACTOR)
BASE_IMG=pygame.transform.scale_by(pygame.image.load(r"E:\pycharm_all_projects\flappy_bird_neat\imgs_b286d95d6d\imgs\BASE.png"),FACTOR)
BG_IMG=pygame.transform.scale_by(pygame.image.load(r"E:\pycharm_all_projects\flappy_bird_neat\imgs_b286d95d6d\imgs\bg.png"),FACTOR)

class Bird:
    IMG=BIRD_IMGS
    MAX_ROTATION=25
    ROT_VEL=25
    ANIMATION_TIME=5
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.tilt=0
        self.tick_count=0
        self.velocity=0
        self.height=self.y
        self.img_count=0
        self.img=self.IMG[0]
    def jump(self):
        self.velocity=-10.5
        self.tick_count=0
        self.height=self.y
        # self.img=self.IMG[1]
        # pygame.time.wait(20)
        # self.img=self.IMG[2]
    def move(self):
        self.tick_count+=1
        d=self.velocity*self.tick_count+1.5*(self.tick_count)**2
        self.y+=d
        if d<0 or self.y<self.height+50:   #moving up or significantly higher than the position from which we earlier jumped
            if self.tilt<self.MAX_ROTATION:
                self.tilt=self.MAX_ROTATION
        else:
            if self.tilt>-90:
                self.tilt-=self.ROT_VEL
    def draw(self,window): #animation
        self.img_count+=1
        #animation
        if self.img_count<self.ANIMATION_TIME:
            self.img=self.IMG[0]
        elif self.img_count<self.ANIMATION_TIME*2:
            self.img=self.IMG[1]
        elif self.img_count<self.ANIMATION_TIME*3:
            self.img=self.IMG[2]
        elif self.img_count<self.ANIMATION_TIME*4:
            self.img=self.IMG[1]
        elif self.img_count==self.ANIMATION_TIME*4+1:
            self.img=self.IMG[0]
            self.img_count=0
        if self.tilt<-80:
            self.img=self.IMG[1]
            self.img_count=self.ANIMATION_TIME*2
        rotated_image=pygame.transform.rotate(self.img,self.tilt)
        new_rect=rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x,self.y)).center)
        window.blit(rotated_image,new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP=100*FACTOR
    def __init__(self,x,vel):
        self.x=x
        self.vel=vel
        self.gap=self.GAP
        self.height=0

        self.top=0
        self.bottom=0
        self.TOP_PIPE=pygame.transform.flip(PIPE_IMG,False,True)
        self.BOTTOM_PIPE=PIPE_IMG
        self.set_height()
        self.passed=False

    def set_height(self):
        self.height=random.randrange(50,450)
        self.top=self.height-self.TOP_PIPE.get_height()
        self.bottom=self.height+self.gap

    def move(self):
        self.x-=self.vel

    def draw(self,window):
        window.blit(self.TOP_PIPE,(self.x,self.top))
        window.blit(self.BOTTOM_PIPE,(self.x,self.bottom))

    def collide(self,bird):
        bird_mask=bird.get_mask()
        top_mask=pygame.mask.from_surface(self.TOP_PIPE)
        bottom_mask=pygame.mask.from_surface(self.BOTTOM_PIPE)

        top_offset=(round(self.x-bird.x),round(self.top-bird.y))
        bottom_offset=(round(self.x-bird.x),round(self.bottom-bird.y))
        top_point=bird_mask.overlap(top_mask,top_offset)
        bottom_point=bird_mask.overlap(bottom_mask,bottom_offset)
        if top_point or bottom_point:
            return True
        return False

class Base:
    VEL=5
    WIDTH=BASE_IMG.get_width()
    IMG=BASE_IMG
    def __init__(self,y):
        self.y=y
        self.vel=self.VEL
        self.x1=0
        self.x2=self.WIDTH
        self.width=self.WIDTH
    def move(self):
        self.x1-=self.vel
        self.x2-=self.vel
        if self.x1+self.width<0:
            self.x1=self.x2+self.width
        if self.x2+self.width<0:
            self.x2=self.x1+self.width
    def draw(self,window):
        window.blit(self.IMG,(self.x1,self.y))
        window.blit(self.IMG,(self.x2,self.y))


def draw_window(window,birds,pipes,base,score):
    window.blit(BG_IMG,(0,0))
    for bird in birds:
        bird.draw(window)
    for pipe in pipes:
        pipe.draw(window)
    text=STAT_FONT.render("Score:"+str(score),1,(255,255,255))
    window.blit(text,(WIN_WIDTH-10-text.get_width(),10))
    base.draw(window)
    pygame.display.update()

def eval_genomes(genomes,config):
    birds=[]
    nets=[]
    ge=[]
    vel = 5
    # bird=Bird(BIRD_X,BIRD_Y)
    base=Base(BASE_Y)
    pipes=[Pipe(500,vel)]
    window=pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    clock=pygame.time.Clock()
    run=True
    score=0
    for _,g in genomes:
        net=neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        g.fitness=0
        ge.append(g)
        birds.append(Bird(BIRD_X,BIRD_Y))
    while run:
        clock.tick(FRAME_RATE)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False
                pygame.quit()
                quit()
        draw_window(window,birds,pipes,base,score)
        # bird.move()
        base.move()
        rem=[]
        bird_rem=[]
        ge_rem=[]
        net_rem=[]
        add_pipe=False
        pipe_ind=0
        if len(birds)>0:
            if len(pipes)>1 and birds[0].x>pipes[0].x+pipes[0].TOP_PIPE.get_width():
                pipe_ind=1
        else:
            run = False
        #print(len(birds), len(ge), len(nets))
        for i in range(len(birds)):
            birds[i].move()
            ge[i].fitness+=0.1
            output=nets[i].activate((birds[i].y,abs(birds[i].y-pipes[pipe_ind].height),abs(birds[i].y-pipes[pipe_ind].bottom),vel))
            if output[0]>0:
                birds[i].jump()
        for pipe in pipes:
            pipe.move()
            for i in range(len(birds)):
                if pipe.collide(birds[i]):
                    bird_rem.append(birds[i])
                    ge_rem.append(ge[i])
                    net_rem.append(nets[i])
                if not pipe.passed and pipe.x<birds[i].x:
                    pipe.passed=True
                    add_pipe=True
            if pipe.x+pipe.TOP_PIPE.get_width()<0:
                rem.append(pipe)

        if add_pipe:
            pipes.append(Pipe(700,vel))
            score += 1
            for g in ge:
                g.fitness+=5
        pipes[0].vel=vel
        for pipe in rem:
            pipes.remove(pipe)
        for i in range(len(birds)):
            if birds[i].y+birds[i].img.get_height()>BASE_Y:
                bird_rem.append(birds[i])
                ge_rem.append(ge[i])
                net_rem.append(nets[i])
            if birds[i].y<0:
                bird_rem.append(birds[i])
                ge_rem.append(ge[i])
                net_rem.append(nets[i])
        flagged=[]
        for i in range(len(bird_rem)):
            print("length of birds:",len(birds))
            print("length of bird_rem:", len(bird_rem))
            if bird_rem[i] in birds:
                print("yes")
            else:
                print("no")
            if bird_rem[i] in birds:
                birds.remove(bird_rem[i])
            else:
                flagged.append(i)
        for i in range(len(ge_rem)):
            if i not in flagged:
                ge.remove(ge_rem[i])
                ge_rem[i].fitness-=1
        for i in range(len(net_rem)):
            if i not in flagged:
                nets.remove(net_rem[i])
        if vel<=20:
            vel+=0.01
        else:
            vel=20
        print(vel)
# main()
def run(config_file):
    config=neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_file)
    p=neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats=neat.StatisticsReporter()
    p.add_reporter(stats)
    winner=p.run(eval_genomes,80)
    with open('winner.pkl', 'wb') as file:
        pickle.dump(winner, file)

if __name__=="__main__":
    local_dir=os.path.dirname(__file__)
    config_path=os.path.join(local_dir,"config.txt")
    run(config_path)