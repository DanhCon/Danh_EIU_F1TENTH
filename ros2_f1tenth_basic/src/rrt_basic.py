import math
import random
import matplotlib.pyplot as plt

class Node:
    def __init__(self,x,y):
        self.x = x
        self.y = y

        self.parent = None

def dist(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def get_nearest_node_index(node_list,rnd_node):
    min_dist = float("inf")
    min_index = -1
    for i in range(len(node_list)):
        node = node_list[i]
        d = dist(node,rnd_node)
        if d < min_dist:
            min_dist =d 
            min_index = i
    return min_index

def steer(from_node, to_node,expand_dis = 3.0):
    dx = to_node.x -from_node.x
    dy = to_node.y - from_node.y
    d = math.sqrt(dx**2 + dy**2)
    theta = math.atan2(dy,dx)

    if d < expand_dis:
        new_node = Node(to_node.x, to_node.y)
    else:
        new_x = from_node.x + expand_dis *math.cos(theta)
        new_y = from_node.y + expand_dis *math.sin(theta)
        new_node = Node(new_x, new_y)
    new_node.parent = from_node
    return new_node

def check_collision(node1 , node2, obstacle_list):
    p1 = [node1.x, node1.y]
    p2 = [node2.x, node2.y]

    steps = int(dist(node1,node2) / 0.5) +1
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    for i in range(steps + 1):
        t = i /steps
        check_x = p1[0] + t*dx 
        check_y = p1[1] +t*dy
        for (ox,oy, size ) in obstacle_list:
            if (check_x - ox)**2 + (check_y -oy )**2 <= size**2:
                return False
    return True

def generate_final_course(goal_node):
    path = [[goal_node.x, goal_node.y]]
    node = goal_node

    while node.parent is not None:
        node = node.parent # Nhảy về cha
        path.append([node.x, node.y])
    
    return path

def main():
    print("da khoi dong")
    map_size = 100
    start = Node(10, 10)
    goal = Node(90, 90) # Mục tiêu cần đến
    
    obstacle_list = [
        (40, 40, 15), (20, 60, 10), (70, 30, 10), (60, 70, 10)
    ]
    
    node_list = [start]
    max_iter = 2000
    step_size = 3.0
    
    path_found = None # Biến lưu đường đi cuối cùng
    count = 0
    max_safety_iter = 10000

    while count < max_safety_iter:
        count = count +1
        rnd = Node(random.uniform(0, map_size), random.uniform(0, map_size))
    
        # B. Tìm cha gần nhất
        nearest_ind = get_nearest_node_index(node_list, rnd)
        nearest_node = node_list[nearest_ind]
        
        # C. Lái về phía đó
        new_node = steer(nearest_node, rnd, expand_dis=step_size)
        
        # D. Check va chạm
        if check_collision(nearest_node, new_node, obstacle_list):
            node_list.append(new_node)
            if dist(new_node, goal) <= step_size:
                print(f"--> Tìm thấy đích ở bước {count}!")
                # Nối node cuối cùng vào goal để đường đẹp hơn
                goal.parent = new_node
                # Gọi hàm truy vết
                path_found = generate_final_course(goal)
                break # Dừng vòng lặp

    # --- VẼ HÌNH ---
    plt.figure(figsize=(8, 8))
    plt.xlim(0, map_size); plt.ylim(0, map_size)

    # Vẽ vật cản
    for (ox, oy, size) in obstacle_list:
        plt.gca().add_patch(plt.Circle((ox, oy), size, color='black'))

    # Vẽ cây (tất cả các node)
    for node in node_list:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", linewidth=0.3)

    # Vẽ đường đi tìm được (Màu Đỏ, Đậm)
    if path_found:
        plt.plot([x for (x, y) in path_found], [y for (x, y) in path_found], '-r', linewidth=2.5, label="Final Path")
        print(f"count{count}")
    else:
        print("Không tìm thấy đường!")

    plt.plot(start.x, start.y, "^b", markersize=10, label="Start")
    plt.plot(goal.x, goal.y, "^r", markersize=10, label="Goal")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()