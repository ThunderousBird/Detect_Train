import subprocess

def generate_requirements(output_file='requirements.txt'):
    # 使用 pip freeze 获取当前环境中的已安装包及其版本号
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
    requirements = result.stdout.decode('utf-8').splitlines()

    # 将包信息写入 requirements.txt 文件
    with open(output_file, 'w') as f:
        for req in requirements:
            f.write(req + '\n')
    
    print(f"Requirements have been generated and saved to {output_file}.")

if __name__ == "__main__":
    generate_requirements()
