import { Message, UserData } from "@/data/data";
import { cn } from "@/lib/utils";
import React, { useRef } from "react";
import { Avatar, AvatarImage } from "../ui/avatar";
import ChatBottombar from "./chat-bottombar";
import { AnimatePresence, motion } from "framer-motion";

interface ChatListProps {
  messages?: Message[];
  selectedUser: UserData;
  // sendMessage: (newMessage: Message) => void;
  isMobile: boolean;
  chatBoxText: string;
}

export function ChatList({
  messages,
  selectedUser,
  // sendMessage,
  isMobile,
  chatBoxText,
}: ChatListProps) {
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="w-full overflow-y-auto overflow-x-hidden h-full flex flex-col ">
      <div
        ref={messagesContainerRef}
        key={messages?.length}
        className="w-full overflow-y-auto overflow-x-hidden h-full flex flex-col"
      >
        <AnimatePresence>
          {messages?.map((message, index) =>
            index !== messages.length - 1 ? (
              <div
                key={index}
                className={cn(
                  "flex flex-col gap-2 p-4 whitespace-pre-wrap",
                  index % 2 !== 0 ? "items-end" : "items-start",
                )}
              >
                <div className="flex gap-3 items-center">
                  {index % 2 === 0 && (
                    <Avatar className="flex justify-center items-center">
                      <AvatarImage
                        src={"/me.webp"}
                        alt={"Me"}
                        width={6}
                        height={6}
                      />
                    </Avatar>
                  )}
                  <span className=" bg-accent p-3 rounded-md max-w-xs">
                    {message.message}
                    {index % 2 !== 0 && (
                      <audio controls muted>
                        <source src={message.audio} type="audio/wav" />
                        Your browser does not support the audio element.
                      </audio>
                    )}
                  </span>
                  {index % 2 !== 0 && (
                    <Avatar className="flex justify-center items-center">
                      <AvatarImage
                        src={"/robo.png"}
                        alt={"AI"}
                        width={6}
                        height={6}
                      />
                    </Avatar>
                  )}
                </div>
              </div>
            ) : (
              <motion.div
                key={index}
                layout
                initial={{ opacity: 0, scale: 1, y: 50, x: 0 }}
                animate={{ opacity: 1, scale: 1, y: 0, x: 0 }}
                exit={{ opacity: 0, scale: 1, y: 1, x: 0 }}
                transition={{
                  opacity: { duration: 0.1 },
                  layout: {
                    type: "spring",
                    bounce: 0.3,
                    duration: messages.indexOf(message) * 0.05 + 0.2,
                  },
                }}
                style={{
                  originX: 0.5,
                  originY: 0.5,
                }}
                className={cn(
                  "flex flex-col gap-2 p-4 whitespace-pre-wrap",
                  index % 2 !== 0 ? "items-end" : "items-start",
                )}
              >
                <div className="flex gap-3 items-center">
                  {index % 2 === 0 && (
                    <Avatar className="flex justify-center items-center">
                      <AvatarImage
                        src={"/me.webp"}
                        alt={"Me"}
                        width={6}
                        height={6}
                      />
                    </Avatar>
                  )}
                  <span className=" bg-accent p-3 rounded-md max-w-xs">
                    {message.message}
                    {index % 2 !== 0 && (
                      <audio controls autoPlay>
                        <source src={message.audio} type="audio/wav" />
                        Your browser does not support the audio element.
                      </audio>
                    )}
                  </span>
                  {index % 2 !== 0 && (
                    <Avatar className="flex justify-center items-center">
                      <AvatarImage
                        src={"/robo.png"}
                        alt={"AI"}
                        width={6}
                        height={6}
                      />
                    </Avatar>
                  )}
                </div>
              </motion.div>
            ),
          )}
        </AnimatePresence>
      </div>
      <ChatBottombar
        // sendMessage={sendMessage}
        isMobile={isMobile}
        chatBoxText={chatBoxText}
      />
    </div>
  );
}
